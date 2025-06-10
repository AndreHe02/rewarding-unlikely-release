# Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening

**Authors**: Andre He, Daniel Fried, Sean Welleck

This repository contains the official implementation for the paper **"Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening"**. 

This codebase is based on [veRL](https://github.com/volcengine/verl) and is designed for RL training of Lean-based theorem-proving models. The Lean verifier is integrated into the online RL loop to provide rewards for model-generated proofs.

---

## Getting Started

### 1. Environment Setup  
Follow the installation and setup instructions provided in the [veRL repository](https://github.com/volcengine/verl) to initialize your environment.

### 2. Lean and Verifier Dependencies  
This project uses a verifier adapted from [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5). To get started:

- Clone and install the DeepSeek-Prover-V1.5 repository.
- Either install it globally or update the path to the verifier in [`verl/lean/verifier.py`](verl/lean/verifier.py).
- Install the [Lean REPL](https://github.com/leanprover-community/repl), which is required to interact with the Lean environment during proof checking.

---

## Running Experiments

- To launch RL training using GRPO, use the script:  
  [`examples/lean/grpo.sh`](examples/lean/grpo.sh)

- The core training loop and implementation of unlikeliness reward are in [`verl/trainer/ppo/ray_lean_trainer.py`](verl/trainer/ppo/ray_lean_trainer.py).

- To evaluate trained checkpoints via sampling:  
  [`examples/inference/run_sampling.sh`](examples/inference/run_sampling.sh)

- To analyze results, compute pass rates, and generate performance plots, see:  
  [`misc/paper_figures.ipynb`](misc/paper_figures.ipynb)

---

## Datasets

To train on your own collection of Lean theorems, you must first convert your dataset into a `.parquet` file. Examples of dataset formats and splits can be found in the [`data`](data/) directory:

- `minif2f_train` and `minif2f_test`: Standard splits of MiniF2F.
- `mff-lwb-goedel-28k.parquet`: A concatenation of `minif2f_train` with ~30k problems from [Lean-Workbook](https://arxiv.org/abs/2406.03847) that were successfully solved by [Godel-Prover](https://goedel-lm.github.io/), excluding problems held out for validation.
- `mff-lwb-10k-seen`: A similarly constructed dataset using theorems solved by [InternLM2.5-step-prover](https://huggingface.co/internlm/internlm2_5-step-prover).

These datasets do not include proofs found by prior work; they are only used to identify provable theorems.
