#!/bin/bash
#SBATCH --job-name=proof-search
#SBATCH --output=logs/eval/proof-search_%A/device_%a.out.txt
#SBATCH --error=logs/eval/proof-search_%A/device_%a.err.txt
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S"
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-13-29,shire-1-1

source ~/.bashrc

# Configuration
NAME=$1

# MODEL=deepseek-ai/DeepSeek-Prover-V1.5-SFT
MODEL=$2

# DATASET=~/projects/verl/data/minif2f_test.parquet
DATASET=$3

OUTPUT_DIR=/data/user_data/awhe/model-evals/${NAME}
NUM_SAMPLES=512
BATCH_SIZE=1

MAX_TOKENS=1024
TEMPERATURE=1.0
MAX_WORKERS=12
PROMPT_KEY=deepseek-prover 

export TMPDIR=/data/user_data/awhe/tmp/

conda activate verl
TOKENIZERS_PARALLELISM=false python examples/inference/proof_sampler.py \
    --dataset ${DATASET} \
    --prompt_key ${PROMPT_KEY} \
    --output_dir ${OUTPUT_DIR} \
    --device_idx ${SLURM_ARRAY_TASK_ID} \
    --num_devices ${SLURM_ARRAY_TASK_COUNT} \
    --max_workers ${MAX_WORKERS} \
    --num_samples ${NUM_SAMPLES} \
    --model_name ${MODEL} \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --batch_size ${BATCH_SIZE}

# call this like
# sbatch --array=0-9 examples/inference/run_sampling.sh base_test_512 model_path dataset_path