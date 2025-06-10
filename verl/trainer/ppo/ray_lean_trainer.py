# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import uuid
import math
import pandas as pd
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import List, Optional, Type, Dict
from collections import defaultdict
import jsonlines

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, collate_fn
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import log_gpu_memory_usage

import torch
from verl.utils.torch_functional import masked_mean

def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

def reduce_metrics_list(metrics_list: List[dict]):
    reduced_metrics = {}
    for key in metrics_list[0]:
        reduced_metrics[key] = np.mean([metrics[key] for metrics in metrics_list])
    return reduced_metrics

def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_data_metrics(batch):
    advantages = batch.batch["advantages"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)

    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(
            torch.eq(prompt_length, max_prompt_length).float()
        )
        .detach()
        .item(),
    }
    return metrics

def compute_timing_metrics(timing_raw):
    return {
        f"timing/{name}": value for name, value in timing_raw.items()
    }

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float] = None):
    with Timer(name=name, logger=None) as timer:
        yield
    if timing_raw is not None:
        if name not in timing_raw:
            timing_raw[name] = timer.last
        else:
            timing_raw[name] += timer.last

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.lean.prompts import get_prompt_fn, get_terminal_token, get_response_fn
from verl.lean.verifier import verify_with_deepseek_verifier
import verl.lean.utils as lean_utils
from verl.lean.utils import compute_pass_metrics, compute_response_metrics

class RayLeanTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        reward_fn,
        val_reward_fn,
    ):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn,
            val_reward_fn,
        )

        if self.config.actor_rollout_ref.actor.update_rule == "sft":
            self.use_reference_policy = False
            self.compute_old_log_probs = False
        else:
            self.use_reference_policy = True
            self.compute_old_log_probs = True

        self.prompt_fn = get_prompt_fn(self.config.lean.prompt_key)
        self.response_fn = get_response_fn(self.config.lean.prompt_key)
        self.terminal_token = get_terminal_token(self.config.lean.prompt_key)
        
        if not hasattr(self, "lean_proofs"):
            self.lean_proofs = []

    def _validate_config(self):
        super()._validate_config()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Set random seed for reproducibility
        seed = self.config.data.get("seed", 42)
        
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.lean.problem_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed),
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.lean.problem_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed),
        )
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps
        
        # we reload proofs so that metrics are resumed
        if getattr(self.config.trainer, "resume", False):
            _, resumed_step, latest_checkpoint = self._find_latest_checkpoint()
            if resumed_step > 0 and latest_checkpoint is not None:
                proofs_dir = os.path.join(self.config.trainer.default_local_dir, "proofs")
                proofs_file = os.path.join(proofs_dir, f"{latest_checkpoint}.jsonl")
                if os.path.exists(proofs_file):
                    print(f"Resuming proofs from checkpoint {proofs_file}")
                    self.lean_proofs = []
                    with jsonlines.open(proofs_file, mode='r') as reader:
                        for item in reader:
                            self.lean_proofs.append(item)
                else:
                    print(f"No proofs found in checkpoint {proofs_file}")

        self.dataset_df = pd.read_parquet(self.config.data.train_files)


    def init_workers(self):
        # super().init_workers() handles loading model checkpoint
        # and sets self.resumed_step
        super().init_workers()

        # now we load the train_batch_buffer
        if (hasattr(self, "resumed_step") and self.resumed_step > 0):
            buffer_dir_name = "train_batch_buffer"
            buffer_local_path = os.path.join(
                self.config.trainer.default_local_dir,
                buffer_dir_name,
                f"global_step_{self.resumed_step}",
            )
            self._load_train_batch_buffer(buffer_local_path)
        
        # we can use these flags to force the run to use some existing buffer
        # and skip directly to sft (for expert iteration)
        if self.config.trainer.get("resume_train_batch_buffer", None) is not None:
            buffer_local_path = self.config.trainer.resume_train_batch_buffer
            self._load_train_batch_buffer(buffer_local_path)

        if self.config.trainer.get("override_resume_step", None) is not None:
            self.resumed_step = self.config.trainer.override_resume_step


    def _load_train_batch_buffer(self, local_path: str):
        try:
            buffer_dataproto = DataProto.load_from_disk(local_path)
            print(f"Loading train batch buffer from {local_path}")
            buffer_dataproto.print_size(prefix="Buffer size")
            self.train_batch_buffer = [buffer_dataproto]
        except FileNotFoundError:
            print(f"No train batch buffer found at {local_path}")
            self.train_batch_buffer = []


    def _generate_and_verify_full_proofs(
        self, problem_batch: DataProto, num_samples: int, timing_raw: dict = None
    ):
        """Generate and verify proofs for a batch of problems in parallel.
        This is much faster than generating in many smaller batches.
        
        Args:
            problem_batch: DataProto containing the problems to solve
            num_samples: Number of proof samples to generate per problem
            timing_raw: Optional dictionary to store timing information
            
        Returns:
            tuple: (train_data, metrics)
                - train_data: DataProto containing generated proofs and metadata
                - metrics: Dictionary with statistics about the generation process
        """
        # Extract problem information
        problem_infos = problem_batch.non_tensor_batch
        theorem_statements = problem_infos["theorem_statement"]
        theorem_full_names = problem_infos["theorem_full_name"]
        informal_statements = problem_infos.get("informal_statement", [""] * len(theorem_statements))
        contexts = problem_infos["src_context"]

        # Generate prompts for all problems
        prompts = []
        for theorem, informal in zip(theorem_statements, informal_statements):
            prompt = self.prompt_fn(theorem, informal, None, None)
            prompts.extend([prompt] * num_samples)

        # Generate proofs
        with _timer("generate_full_proof", timing_raw):
            gen_batch = self._format_model_input(prompts)
            gen_tensors = self.actor_rollout_wg.generate_sequences(gen_batch)
            responses = gen_tensors.batch["responses"]
            proofs = [self.tokenizer.decode(r, skip_special_tokens=True) for r in responses]

        # Log sample proofs
        print(f"[SAMPLING] Generated proof:")
        print(proofs[0])
        print(f"[SAMPLING] Generated proof:")
        print(proofs[-1])

        # Parse proofs into steps
        proof_steps = []
        for proof in proofs:
            try:
                steps = lean_utils.parse_proof_steps(
                    proof, 
                    retain_comments=self.config.lean.get("retain_comments", False),
                    terminal_token=self.terminal_token,
                )
                proof_steps.append(steps)
            except Exception as e:
                print(f"[SAMPLING] Failed to parse proof: {e}")
                proof_steps.append([])

        # Verify proofs
        with _timer("verify_full_proof", timing_raw):
            outputs = verify_with_deepseek_verifier(
                proofs,
                theorem_statements,
                contexts,
                max_workers=self.config.lean.max_workers,
                penalize_extra_text=self.config.lean.get("penalize_extra_text", False),
            )

            # Log verification results
            for i, out in enumerate(outputs):
                print(f"[SAMPLING] {theorem_full_names[i]}")
                print(f"[SAMPLING] {out['num_success']} / {num_samples}")
                print(f"[SAMPLING] {out['msg']}\n")

        # Store Lean proofs
        for i in range(len(proofs)):
            problem_idx = i // num_samples
            theorem_full_name = theorem_full_names[problem_idx]
            theorem_statement = theorem_statements[problem_idx]
            informal_statement = informal_statements[problem_idx]
            context = contexts[problem_idx]
            out = outputs[problem_idx]
            is_correct = (i % num_samples) in out["success_indices"]
            proof = proofs[i]
            
            self.lean_proofs.append(lean_utils.make_lean_proof(
                theorem_full_name,
                theorem_statement,
                informal_statement,
                proof,
                correct=is_correct,
                context=context,
            ))

        # Build training data
        selected_idxs = []
        rewards = []
        metadata = defaultdict(list)
        all_meta_keys = problem_batch[0].non_tensor_batch.keys()
        for k in all_meta_keys:
            metadata[k] = []
        
        for i, out in enumerate(outputs):
            problem_metadata = problem_batch[i].non_tensor_batch
            num_samples_from_problem = 0
            
            for j in range(num_samples):
                reward = 1.0 if j in out["success_indices"] else 0.0
                has_adv = out["num_success"] > 0 and out["num_success"] < num_samples
            
                # Apply data filters
                if self.config.lean.rejection_sampling and reward == 0.0:
                    continue

                if self.config.lean.advantage_threshold and not has_adv:
                    continue

                if num_samples_from_problem >= self.config.lean.max_samples:
                    continue

                # Add sample to training data
                selected_idxs.append(i * num_samples + j)
                rewards.append(reward)
                for key, value in problem_metadata.items():
                    metadata[key].append(value)
                num_samples_from_problem += 1

        # Extract selected tensors and proofs
        gen_tensors = gen_tensors.select_indices(selected_idxs)
        filtered_proofs = [proofs[i] for i in selected_idxs]
        
        # Build extra info DataProto
        extra_info = {
            "full_proofs": np.array(filtered_proofs),
            "rewards": torch.tensor(rewards),
            **{key: np.array(value) for key, value in metadata.items()},
        }
        extra_info = DataProto.from_single_dict(extra_info)
        train_data = gen_tensors.union(extra_info)

        # Check for truncated responses
        response_shape = gen_tensors.batch["responses"].size(-1)
        response_attention_mask = gen_tensors.batch["attention_mask"][:, -response_shape:]
        response_lengths = response_attention_mask.sum(-1)
        truncated = torch.eq(response_lengths, response_shape)
        
        # Add truncation info to training data
        train_data = train_data.union(
            DataProto.from_single_dict({"truncated": truncated})
        )
        num_truncated = truncated.sum()
        print(f"[TRAINING] Truncated: {num_truncated}")

        # Prepare metrics
        metrics = {
            "num_accepted": len(filtered_proofs),
            "num_rejected": len(proofs) - len(filtered_proofs),
            "num_errors": sum(out["num_errors"] for out in outputs),
            "num_truncated": num_truncated,
        }

        # compute batch metrics
        total_reward = 0.0
        total_solved = 0
        for out in outputs:
            total_reward += out["num_success"]
            total_solved += int(out["num_success"] > 0)
        
        metrics["batch_reward"] = total_reward / len(proofs)
        metrics["batch_pass"] = total_solved / len(outputs)

        return train_data, metrics


    def _format_model_input(self, prompts: List[str]):
        # we should refer to vLLMRollouts.generate sequences for how to construct the input batch
        input_ids = []
        attention_masks = []
        position_ids = []
        for i, prompt in enumerate(prompts):
            input_id, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=self.tokenizer,   
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            input_ids.append(input_id[0])
            attention_masks.append(attention_mask[0])
            position_ids.append(compute_position_id_with_mask(attention_mask)[0])

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        position_ids = torch.stack(position_ids)

        input_batch = DataProto.from_dict(
            dict(
                input_ids=input_ids,
                attention_mask=attention_masks,
                position_ids=position_ids,
            )
        )
        return input_batch

    def _save_proofs(self):
        local_path = os.path.join(
            self.config.trainer.default_local_dir,
            "proofs",
            f"global_step_{self.global_steps}.jsonl",
        )
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # Convert defaultdicts to regular dicts for JSON serialization

        with jsonlines.open(local_path, mode='w') as writer:
            for proof in self.lean_proofs:
                writer.write(proof)

        # Clean up old checkpoints if save_limit is set
        save_limit = self.config.trainer.get("save_limit", None)
        if save_limit is not None and save_limit > 0:
            # Handle actor checkpoints
            proofs_dir = os.path.join(self.config.trainer.default_local_dir, "proofs")
            if os.path.exists(proofs_dir):
                checkpoints = sorted([
                    d for d in os.listdir(proofs_dir) 
                    if os.path.isdir(os.path.join(proofs_dir, d)) and d.startswith("global_step_")
                ], key=lambda x: int(x.split("_")[-1]))
                
                # Remove oldest checkpoints if we exceed the limit
                while len(checkpoints) > save_limit:
                    oldest_checkpoint = checkpoints.pop(0)
                    oldest_path = os.path.join(proofs_dir, oldest_checkpoint)
                    print(f"Removing old proofs checkpoint: {oldest_path}")
                    shutil.rmtree(oldest_path)

    def _save_checkpoint(self):
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir,
            "actor",
            f"global_step_{self.global_steps}",
        )
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "actor")
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            raise NotImplementedError
        
        # Clean up old checkpoints if save_limit is set
        save_limit = self.config.trainer.get("save_limit", None)
        if save_limit is not None and save_limit > 0:
            # Handle actor checkpoints
            actor_dir = os.path.join(self.config.trainer.default_local_dir, "actor")
            if os.path.exists(actor_dir):
                checkpoints = sorted([
                    d for d in os.listdir(actor_dir) 
                    if os.path.isdir(os.path.join(actor_dir, d)) and d.startswith("global_step_")
                ], key=lambda x: int(x.split("_")[-1]))
                
                # Remove oldest checkpoints if we exceed the limit
                while len(checkpoints) > save_limit:
                    oldest_checkpoint = checkpoints.pop(0)
                    oldest_path = os.path.join(actor_dir, oldest_checkpoint)
                    print(f"Removing old actor checkpoint: {oldest_path}")
                    shutil.rmtree(oldest_path)

        # we also need to checkpoint the train_batch_buffer
        # self.train_batch_buffer is a list of DataProto
        buffer_dir_name = "train_batch_buffer"
        buffer_local_path = os.path.join(
            self.config.trainer.default_local_dir,
            buffer_dir_name,
            f"global_step_{self.global_steps}",
        )
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(buffer_local_path), exist_ok=True)
        # we save the buffer as a single DataProto
        # when resuming, take care to make it a list of DataProto
        if len(self.train_batch_buffer) > 0:
            buffer_dataproto = DataProto.concat(self.train_batch_buffer)
            buffer_dataproto.save_to_disk(buffer_local_path)
        else:
            print(f"[TRAINING] No train batch buffer to save")
        
        if save_limit is not None and save_limit > 0:
            buffer_dir = os.path.join(self.config.trainer.default_local_dir, buffer_dir_name)
            if os.path.exists(buffer_dir):
                checkpoints = sorted([
                    d for d in os.listdir(buffer_dir) 
                    if os.path.isdir(os.path.join(buffer_dir, d)) and d.startswith("global_step_")
                ], key=lambda x: int(x.split("_")[-1]))
                
                # Remove oldest checkpoints if we exceed the limit
                while len(checkpoints) > save_limit:
                    oldest_checkpoint = checkpoints.pop(0)
                    oldest_path = os.path.join(buffer_dir, oldest_checkpoint)
                    print(f"Removing old train batch buffer checkpoint: {oldest_path}")
                    shutil.rmtree(oldest_path)


    def _should_update_now(self, is_epoch_end: bool):
        if self.config.trainer.get("sample_only", False):
            self.train_batch_buffer = []
            return False
        dynamic_update = self.config.trainer.get("dynamic_update", False)
        if not dynamic_update:    
            update_freq = self.config.trainer.get("update_freq", 1)
            if update_freq == 0:
                return False
            elif update_freq == -1:
                return is_epoch_end
            else:
                return self.global_steps % update_freq == 0
        else:
            buffer_size = sum(len(batch) for batch in self.train_batch_buffer)
            print(f"[TRAINING] Buffer size: {buffer_size}")
            return buffer_size >= self.config.trainer.train_batch_size
        

    def _compute_advantages(self, batch, epsilon=1e-6):
        # assert "old_log_probs" in batch.batch
        scores = batch.batch["rewards"]
        index = batch.non_tensor_batch["uid"]
        old_log_probs = batch.batch["old_log_probs"]

        response_length = batch.batch["responses"].shape[-1]
        eos_mask = batch.batch["attention_mask"][:, -response_length:]
        old_log_probs = masked_mean(old_log_probs, eos_mask, axis=-1)

        # Create a modified_scores tensor to hold our adjusted scores
        modified_scores = scores.clone()
        
        rank_penalty = self.config.lean.get("rank_penalty", 0.0)
    
        with torch.no_grad():
            bsz = scores.shape[0]
            
            # First, collect all indices for each group ID
            id_to_indices = defaultdict(list)
            for i in range(bsz):
                id_to_indices[index[i]].append(i)
            
            # Process each group
            for idx, indices in id_to_indices.items():
                # Get the scores and log probs for this group
                group_indices = torch.tensor(indices, device=scores.device)
                group_scores = scores[group_indices]
                group_old_log_probs = old_log_probs[group_indices]
                
                # Apply rank penalty
                group_ranks = torch.argsort(torch.argsort(group_old_log_probs))
                group_ranks = group_ranks / len(group_indices)
                group_scores = group_scores * (1 - group_ranks * rank_penalty)
                
                # Update the modified_scores for these indices
                modified_scores[group_indices] = group_scores
                
                # Calculate group stats
                if len(group_indices) == 1:
                    group_mean = torch.tensor(0.0, device=scores.device)
                    group_std = torch.tensor(1.0, device=scores.device)
                else:
                    group_mean = torch.mean(group_scores)
                    group_std = torch.std(group_scores)
                
                # Normalize scores for this group
                for i in indices:
                    scores[i] = (modified_scores[i] - group_mean) / (group_std + epsilon)

            # Expand to token level
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        batch.batch["advantages"] = scores
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 1

        if not hasattr(self, "train_batch_buffer"):
            self.train_batch_buffer = []
        else:
            print(f"[TRAINING] Resuming from train buffer from step {self.resumed_step}")

        update_freq = self.config.trainer.get("update_freq", 1)
        
        for epoch in range(self.config.trainer.total_epochs):

            skip_dataloader_steps = self.config.trainer.get("skip_dataloader_steps", 0)

            for batch_idx, batch_dict in enumerate(self.train_dataloader):

                if skip_dataloader_steps > 0:
                    if batch_idx < skip_dataloader_steps:
                        self.global_steps += 1
                        continue

                if hasattr(self, "resumed_step"):
                    # if we resume from a checkpoint, we skip the steps before the checkpoint
                    # resume from global_step N + 1
                    if self.global_steps <= self.resumed_step:
                        print(f"[TRAINING] Skipping step {self.global_steps} <= {self.resumed_step}")
                        self.global_steps += 1
                        continue

                metrics = {}
                timing_raw = {}
                problem_batch: DataProto = DataProto.from_single_dict(batch_dict)
                problem_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(problem_batch.batch))],
                    dtype=object,
                )

                problem_batch, _ = pad_dataproto_to_divisor(problem_batch, self.actor_rollout_wg.world_size)

                with _timer("step", timing_raw):

                    with _timer("generate_and_verify_proofs", timing_raw):
                        batch, verify_metrics = self._generate_and_verify_full_proofs(
                            problem_batch, 
                            self.config.lean.num_samples, 
                            timing_raw=timing_raw
                        )
                                            
                    self.train_batch_buffer.append(batch)                   
                    metrics.update(verify_metrics)
                    print(f"[TRAINING] Step train batch size: {len(batch)}")
                    metrics.update({
                        "train/buffer_size": sum(len(batch) for batch in self.train_batch_buffer)
                    })

                    if self._should_update_now(is_epoch_end=(batch_idx == len(self.train_dataloader) - 1)):
                        batch = DataProto.concat(self.train_batch_buffer)
                        self.train_batch_buffer = []
                        print(f"[TRAINING] Update train batch size: {len(batch)}")
                        
                        # batch = self._format_train_batch_v2(batch)
                        batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # recompute old_log_probs
                        # we don't need to compute this for rft
                        if self.compute_old_log_probs:
                            with _timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                                batch = batch.union(ref_log_prob)

                        if self.config.actor_rollout_ref.actor.update_rule == "ppo":
                            with _timer("compute_advantages", timing_raw):
                                batch = self._compute_advantages(batch)

                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(
                                actor_output.meta_info["metrics"]
                            )
                            pprint(actor_output_metrics)
                            metrics.update(actor_output_metrics)

                        metrics.update(compute_data_metrics(batch=batch))

                if (
                    self.config.trainer.save_freq > 0
                    and self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                if (
                    self.config.trainer.save_proof_freq > 0 
                    and self.global_steps % self.config.trainer.save_proof_freq == 0
                ):
                    with _timer("save_proofs", timing_raw):
                        self._save_proofs()

                metrics.update(compute_pass_metrics(self.dataset_df, self.lean_proofs, self.config.lean.num_samples, prefix="train"))
                metrics.update(compute_response_metrics(self.dataset_df, self.lean_proofs, prefix="train"))
                metrics.update(compute_timing_metrics(timing_raw=timing_raw))

                pprint(f"Metrics: {metrics}")
                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.total_training_steps:
                    print(f"[TRAINING] Training finished")
                    raise Exception("Stop")
                    return

                self.global_steps += 1

            # this section is only for expert iteration
            # it is a separte update loop that goes through the train buffer and updates the actor
            if self.config.trainer.get("expert_iter", False):

                # I have no idea why this is needed to prevent OOM
                dummy_batch = self._format_model_input(["hello" for _ in range(8)])
                dummy_output = self.actor_rollout_wg.generate_sequences(dummy_batch)

                print(f"[TRAINING] Expert iteration update")
                # checkpoint before expert iteration
                self._save_checkpoint()
                self._save_proofs()

                sft_batch_size = self.config.trainer.get("sft_batch_size", 256)
                sft_data = DataProto.concat(self.train_batch_buffer)
                self.train_batch_buffer = [] # don't do this if accumulating data
                sft_data = self._format_train_batch(sft_data)
                sft_data, pad_size = pad_dataproto_to_divisor(sft_data, self.actor_rollout_wg.world_size)
                sft_data, pad_size = pad_dataproto_to_divisor(sft_data, sft_batch_size)
                
                for sft_epoch in range(self.config.trainer.get("sft_epochs", 1)):
                    sft_data.shuffle(seed=self.global_steps)
                    sft_batches = sft_data.chunk(chunks=len(sft_data) // sft_batch_size)

                    for sft_batch_idx, sft_batch in enumerate(sft_batches):
                        if self.global_steps <= self.resumed_step:
                            print(f"[TRAINING] Skipping step {self.global_steps} <= {self.resumed_step}")
                            self.global_steps += 1
                            continue

                        print(f"[TRAINING] Expert iteration update step {sft_batch_idx}, global_step {self.global_steps}")

                        self._balance_batch(sft_batch, metrics={})

                        sft_batch.meta_info["global_token_num"] = torch.sum(
                            sft_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        actor_output = self.actor_rollout_wg.update_actor(sft_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        pprint(actor_output_metrics)

                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            self._save_checkpoint()

                        logger.log(data=actor_output_metrics, step=self.global_steps)
                        self.global_steps += 1                

            # save every epoch
            # always save every epoch for expert iteration
            if self.config.trainer.save_freq == -1 or self.config.trainer.get("expert_iter", False):
                self._save_checkpoint()

            if self.config.trainer.save_proof_freq == -1 or self.config.trainer.get("expert_iter", False):
                self._save_proofs()

