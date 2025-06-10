#!/bin/bash
#SBATCH -J leanrl                 # Job name
#SBATCH -o logs/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --error=logs/%x_%j.err
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=500G                   # server memory requested (per node)
#SBATCH --time=48:00:00
#SBATCH --partition=general               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40S:4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --exclude=babel-10-5,babel-13-1,babel-13-13,babel-4-13

#comment --nodelist=babel-0-23,babel-0-27,babel-0-31,babel-0-19,babel-a-20,babel-0-37,babel-1-23,babel-1-31,babel-1-27,shire-2-5,shire-2-9,babel-3-21,babel-5-23,babel-5-31,babel-5-27,babel-6-17,babel-6-21,flame-9-21,flame-10-21,babel-11-5,babel-11-13,babel-15-20,babel-15-24

nvidia-smi topo -m

source ~/.bashrc
conda activate verl

MODEL=deepseek-ai/DeepSeek-Prover-V1.5-SFT
PROMPT_KEY=deepseek-prover 
# DATASET=~/projects/verl/data/mff-lwb-10k-seen.parquet
DATASET=~/projects/verl/data/mff-lwb-goedel-28k.parquet
TEST_DATASET=~/projects/verl/data/minif2f_test.parquet
EXPERIMENT_NAME=$1
MAX_WORKERS=64

# turn this on for A6000s or L40s that dont have NVLINK
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# make sure model loads
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_lean \
 data.train_files=${DATASET} \
 data.val_files=${TEST_DATASET} \
 data.max_prompt_length=512 \
 +data.seed=42 \
 actor_rollout_ref.model.path=${MODEL} \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
 actor_rollout_ref.actor.optim.weight_decay=0.01 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 critic.ppo_micro_batch_size=8 \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE} \
 trainer.nnodes=1 \
 trainer.logger=\[console,wandb\] \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.rollout.response_length=512 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.1 \
 actor_rollout_ref.actor.ppo_epochs=2 \
 algorithm.adv_estimator=grpo \
 +actor_rollout_ref.model.trust_remote_code=True \
 actor_rollout_ref.rollout.load_format=dummy_dtensor \
 actor_rollout_ref.actor.update_rule=ppo \
 trainer.experiment_name=${EXPERIMENT_NAME} \
 trainer.project_name=verl \
 trainer.save_freq=100 \
 trainer.save_limit=1 \
 trainer.save_proof_freq=50 \
 trainer.total_epochs=1 \
 lean.prompt_key=${PROMPT_KEY} \
 lean.num_samples=32 \
 lean.problem_batch_size=16 \
 lean.rejection_sampling=False \
 lean.advantage_threshold=True \
 lean.max_workers=${MAX_WORKERS} \
 actor_rollout_ref.actor.entropy_coeff=0.0 \
 actor_rollout_ref.actor.grad_skip_thresh=20.0 \
 trainer.train_batch_size=256 \
 trainer.dynamic_update=True \
 +trainer.resume=True \
 +lean.penalize_extra_text=True \
 +lean.rank_penalty=0.25
#  +trainer.sample_only=True
