# for unknown reasons, we have to import pandas last
import argparse
import os
import json
import torch
import time
import glob
from tqdm import tqdm
# # import transformers
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Set
import jsonlines
import pandas as pd

# Import necessary functions from verl modules
from verl.lean.prompts import get_prompt_fn, get_terminal_token
from verl.lean.verifier import verify_full_proof_batch, verify_with_timeout, verify_with_deepseek_verifier
from verl.lean.utils import parse_proof_steps, make_lean_proof

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sample proofs from a model and verify them")
    parser.add_argument("--dataset", type=str, required=True, help="Parquet file containing theorem statements")
    parser.add_argument("--prompt_key", type=str, required=True, help="Key pointing to a specific prompt format")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--device_idx", type=int, required=True, help="Index of this instance in the job array")
    parser.add_argument("--num_devices", type=int, required=True, help="Total number of jobs in the array")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of workers for verification")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of proofs to sample per problem")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum iterations for verification")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for token handling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--stop_tokens", type=str, default="\ntheorem,\n\n\n,---,[/TAC]", 
                        help="Comma-separated list of stop tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing problems")
    return parser.parse_args()

def _load_model(model_path, tp_degree):
    """Load the model and tokenizer using vLLM"""
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=tp_degree,
        dtype="bfloat16",
        max_logprobs=100,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        swap_space=64,
    )

    return model

# Removed _unique_sorted function since we want to use samples directly

def generate_vllm(
    prompts,
    model,
    temperature,
    num_samples=1,
    max_tokens=2048,
    stop=None,
    batch_size=4,
):
    """Sample from a vLLM model locally"""
    all_texts = []
    
    if num_samples <= 128:
        params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # try:
            outputs = model.generate(batch_prompts, params, use_tqdm=False)
            
            for prompt_outputs in outputs:
                for output in prompt_outputs.outputs:
                    text = output.text
                    all_texts.append(text)
    else:
        generate_batch_size = 128
        params = SamplingParams(
            n=generate_batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=0.90,
        )

        text_batches = []
        for _ in range(num_samples // generate_batch_size):
            print(f"Generating mini batch {len(text_batches) + 1} of {num_samples // generate_batch_size}")
            texts = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                outputs = model.generate(batch_prompts, params, use_tqdm=False)
                for prompt_outputs in outputs:
                    for output in prompt_outputs.outputs:
                        text = output.text
                        texts.append(text)
            text_batches.append(texts)

        # now merge the text batches
        # every 128 proofs of text_batches[i] is from the same prompt
        all_texts = []
        for i in range(len(prompts)):
            for texts in text_batches:
                responses_to_prompt = texts[i*generate_batch_size:(i+1)*generate_batch_size]
                all_texts.extend(responses_to_prompt)
        
    return all_texts

def process_batch(
    problem_batch, 
    prompt_fn, 
    model,
    temperature: float,
    num_samples: int, 
    max_tokens: int,
    max_iters: int, 
    max_workers: int,
    stop_tokens=None,
    terminal_token=None,
    vllm_batch_size=4,
):
    """Process a batch of problems, sample proofs, and verify them"""
    theorem_statements = problem_batch["theorem_statement"].tolist()
    theorem_full_names = problem_batch["theorem_full_name"].tolist()
    contexts = problem_batch["src_context"].tolist()
    
    # Get informal statements if they exist, otherwise use empty strings
    informal_statements = problem_batch.get("informal_statement", [""] * len(theorem_statements))
    if isinstance(informal_statements, pd.Series):
        informal_statements = informal_statements.tolist()
    
    # Generate prompts for all problems
    prompts = []
    for theorem, informal in zip(theorem_statements, informal_statements):
        prompt = prompt_fn(theorem, informal, None, None)
        prompts.append(prompt)
    
    # Sample proofs from the model - this will generate num_samples for each prompt
    proofs = generate_vllm(prompts, model, temperature, num_samples, max_tokens, stop_tokens, batch_size=vllm_batch_size)

    print(f"Generated proof example")
    print(proofs[0])
    
    # Parse proofs into steps
    proof_steps = []
    for proof in proofs:
        try:
            steps = parse_proof_steps(
                proof, 
                retain_comments=False,
                terminal_token=terminal_token,
            )
            proof_steps.append(steps)
        except Exception as e:
            print(f"Failed to parse proof: {e}")
            proof_steps.append([])
    
    # Use the batched version of verify_full_proof
    # todo use the version with hard timeout
    # verification_results, completed = verify_with_timeout(
    #     proof_steps,
    #     proofs,
    #     theorem_statements,
    #     contexts,
    #     max_iters=max_iters,
    #     max_workers=max_workers,
    #     timeout=10000,
    #     return_completed=True,
    # )

    # if not completed:
    #     print("WARNING: verify_with_timeout did not complete within 1000 seconds")
    #     print("Skipping this batch")
    #     return []
    

    verification_results = verify_with_deepseek_verifier(
        proofs,
        theorem_statements,
        contexts,
        max_workers=max_workers,
    )


    # Process results into output format
    results = []
    for i, (theorem, theorem_name, verification) in enumerate(zip(theorem_statements, theorem_full_names, verification_results)):
        problem_proofs = proofs[i*num_samples:(i+1)*num_samples]
        # problem_scores = scores[i*num_samples:(i+1)*num_samples]
        # problem_trace = verification["search_trace"]
        success_indices = verification["success_indices"]
        
        print(f"Theorem: {theorem_name}")
        print(f"Verification success: {verification['success']}")
        print(f"Number of successful proofs: {verification['num_success']}")
        print(f"Number of errors: {verification['num_errors']}")
        print(f"Success indices: {success_indices}")
        
        for j, proof in enumerate(problem_proofs):
            proof_obj = make_lean_proof(
                theorem_name=theorem_name,
                theorem_statement=theorem,
                informal_statement=informal_statements[i],
                proof=proof,
                correct=j in success_indices,
                context=contexts[i],
            )
            results.append(proof_obj)
    
    return results

def get_completed_theorems(output_dir, device_idx):
    """
    Get a set of theorem full names that have already been processed to enable resuming.
    """
    completed_theorems = set()
    
    # Check both the intermediate batch files and the final result file
    batch_file_pattern = os.path.join(output_dir, f"proofs_device{device_idx}_batch*.jsonl")
    final_file = os.path.join(output_dir, f"proofs_device{device_idx}.jsonl")
    
    all_files = glob.glob(batch_file_pattern)
    if os.path.exists(final_file):
        all_files.append(final_file)
    
    for filepath in all_files:
        try:
            with jsonlines.open(filepath, mode='r') as reader:
                for item in reader:
                    completed_theorems.add(item["theorem_name"])
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    print(f"Found {len(completed_theorems)} already processed theorems for device {device_idx}")
    return completed_theorems

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get prompt function and terminal token
    prompt_fn = get_prompt_fn(args.prompt_key)
    terminal_token = get_terminal_token(args.prompt_key)
    
    # Parse stop tokens
    stop_tokens = args.stop_tokens.split(",") if args.stop_tokens else []
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name} with tensor parallel size {args.tensor_parallel_size}")
    model = _load_model(args.model_name, args.tensor_parallel_size)
    print("Model loaded successfully")
    
    # Load dataset
    df = pd.read_parquet(args.dataset)
    
    # Filter problems based on device_idx and num_devices
    total_problems = len(df)
    device_problems = [i for i in range(total_problems) if i % args.num_devices == args.device_idx]
    df_subset = df.iloc[device_problems].reset_index(drop=True)
    
    print(f"Device {args.device_idx} assigned {len(df_subset)} out of {total_problems} problems")
    
    # Check for already processed theorems to enable resuming
    completed_theorems = get_completed_theorems(args.output_dir, args.device_idx)
    
    # Filter out already processed theorems
    if completed_theorems:
        df_subset = df_subset[~df_subset["theorem_full_name"].isin(completed_theorems)].reset_index(drop=True)
        print(f"After filtering out completed theorems, {len(df_subset)} problems remain to be processed")
    
    # Nothing to do if all theorems are already processed
    if len(df_subset) == 0:
        print(f"All problems for device {args.device_idx} have already been processed. Exiting.")
        return
    
    # Process problems in batches to avoid memory issues
    batch_size = args.batch_size  # Small batch size to avoid overwhelming the GPU
    all_results = []

    start_time = time.time()
    
    for i in tqdm(range(0, len(df_subset), batch_size)):
        batch = df_subset.iloc[i:i+batch_size]
        batch_theorems = batch["theorem_full_name"].tolist()
        print(f"Processing batch {i // batch_size + 1}/{(len(df_subset) - 1) // batch_size + 1}: {batch_theorems}")
        
        results = process_batch(
            batch,
            prompt_fn,
            model,
            args.temperature,
            args.num_samples,
            args.max_tokens,
            args.max_iters,
            args.max_workers,
            stop_tokens,
            terminal_token,
            vllm_batch_size=batch_size,
        )
        all_results.extend(results)
        
        # Save intermediate results
        # Find existing batch files and determine the next available batch index
        existing_batch_files = glob.glob(os.path.join(args.output_dir, f"proofs_device{args.device_idx}_batch*.jsonl"))
        existing_indices = []
        for file in existing_batch_files:
            try:
                # Extract the batch index from the filename
                filename = os.path.basename(file)
                idx = int(filename.split("batch")[1].split(".")[0])
                existing_indices.append(idx)
            except (ValueError, IndexError):
                continue
        
        # Use the next available index, or 0 if no files exist
        if len(results) > 0:
            batch_idx = max(existing_indices) + 1 if existing_indices else 0
            batch_output_file = os.path.join(args.output_dir, f"proofs_device{args.device_idx}_batch{batch_idx}.jsonl")
            with jsonlines.open(batch_output_file, mode='w') as writer:
                for result in results:
                    writer.write(result)
                    
            print(f"Saved intermediate results to {batch_output_file}")
        else:
            print(f"No results were generated in this batch")

        curr_time = time.time()
        print(f"Running average: {(curr_time - start_time) / (i + 1)} seconds per batch")

    
    # Combine all intermediate results into the final file if we have any
    if len(all_results) == 0:
        print("No results were generated in this run.")
        return
    
    # We'll read all the batch files to ensure we include any from previous runs
    # that might not be in our current all_results list
    final_results = []
    batch_files = glob.glob(os.path.join(args.output_dir, f"proofs_device{args.device_idx}_batch*.jsonl"))
    
    for file in batch_files:
        try:
            with jsonlines.open(file, mode='r') as reader:
                for item in reader:
                    final_results.append(item)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")
    
    # # Remove duplicates (based on theorem_full_name and sample_idx)
    # seen = set()
    # unique_results = []
    # for result in final_results:
    #     key = (result["theorem_name"], result["sample_idx"])
    #     if key not in seen:
    #         seen.add(key)
    #         unique_results.append(result)
    unique_results = final_results

    # write the final result if all problems have been processed
    if len(unique_results) >= len(df_subset) * args.num_samples:
        print("All problems have been processed, writing final result")
        final_output_file = os.path.join(args.output_dir, f"proofs_device{args.device_idx}.jsonl")
        with jsonlines.open(final_output_file, mode='w') as writer:
            for result in unique_results:
                writer.write(result)
        
        print(f"Saved final combined results to {final_output_file}")
    
    # Calculate and print summary statistics
    num_correct = sum(1 for result in unique_results if result["correct"])
    total_proofs = len(unique_results)
    print(f"Success rate: {num_correct}/{total_proofs} ({num_correct/total_proofs:.2%})")

if __name__ == "__main__":
    main()