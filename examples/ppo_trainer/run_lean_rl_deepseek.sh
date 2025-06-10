#!/bin/bash
#SBATCH -J leanrl                 # Job name
#SBATCH -o logs/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --error=logs/%x_%j.err
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=220G                   # server memory requested (per node)
#SBATCH --time=48:00:00
#SBATCH --partition=general               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:4   
#SBATCH --open-mode=append            # Do not overwrite logs

source ~/.bashrc
conda activate verl

MODEL=deepseek-ai/DeepSeek-Prover-V1.5-RL
# this key is used to identify the prompt template 
MODEL_NAME=deepseek-prover
DATASET=/home/awhe/projects/minictx-eval/notebooks/data/minif2f.parquet
# DATASET=/home/awhe/projects/minictx-eval/notebooks/data/mathd_numbertheory_232.parquet
export NCCL_P2P_DISABLE=1

# make sure model loads
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=${DATASET} \
 data.val_files=${DATASET} \
 data.max_prompt_length=256 \
 actor_rollout_ref.model.path=${MODEL} \
 actor_rollout_ref.actor.optim.lr=1e-5 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 critic.ppo_micro_batch_size=8 \
 algorithm.kl_ctrl.kl_coef=0.00 \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.total_epochs=5 \
 trainer.logger=\[console,wandb\] \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.rollout.response_length=512 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.kl_loss_coef=0.0 \
 actor_rollout_ref.actor.ppo_epochs=2 \
 algorithm.adv_estimator=grpo \
 +actor_rollout_ref.model.trust_remote_code=True \
 actor_rollout_ref.rollout.load_format=dummy_dtensor \
 +trainer.type=lean \
 +lean.model_name=${MODEL_NAME} \
 +lean.num_samples=32 \
 +lean.max_iters=32 \
 +lean.max_goal_len=1024 \
 +lean.val_num_samples=4 \
 +trainer.val_before_train=False \
 trainer.experiment_name=lean_rl \
 trainer.project_name=verl \
 +lean.problem_batch_size=16 \
 +lean.rejection_sampling=True \
 +lean.advantage_threshold=0.0 \
 +lean.merge_states=False \
 +lean.merge_steps=True \
 actor_rollout_ref.actor.update_rule=sft \
 trainer.test_freq=100 \
 +trainer.save_proof_freq=1 \
 +lean.verify_training_proofs=False \
 +lean.batched_inference=True \
 +lean.max_workers=8 \
 +lean.full_proof=True


#  +data.test_files=/home/awhe/projects/minictx-eval/notebooks/data/minif2f_small.parquet \
#--exclude=babel-11-21,babel-1-31,babel-14-1,babel-7-13,babel-15-32,babel-1-23,shire-1-6,babel-5-15,babel-5-11,babel-5-19,babel-4-25,babel-11-9,babel-13-13,shire-1-10,babel-1-31,babel-0-37,babel-1-23,babel-4-25,babel-4-17,babel-6-9
