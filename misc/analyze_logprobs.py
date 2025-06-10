import argparse
import os
import pickle
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append("/home/awhe/projects/verl")
from verl.lean.prompts import prompt_fn_deepseek

import jsonlines

def load_proofs(proofs_file):
    proofs = []
    with jsonlines.open(proofs_file) as reader:
        for obj in reader:
            proofs.append(obj)
    return proofs

def construct_full_text(proof):
    theorem_statement = proof['theorem_statement']
    informal_statement = proof['informal_statement']
    theorem_full_name = proof['theorem_name']

    prompt = prompt_fn_deepseek(
        theorem_statement, informal_statement, theorem_full_name, None, None
    )

    proof_body = proof['proof']
    terminates = "```" in proof_body
    proof_body = proof_body.split("```")[0]
    if terminates:
        proof_body += "```"

    return prompt, proof_body

def get_proof_logprob(proof, model, tokenizer):
    prompt, proof_body = construct_full_text(proof)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    response_ids = tokenizer.encode(proof_body, return_tensors="pt", add_special_tokens=False).to(model.device)
    full_ids = torch.cat([input_ids, response_ids], dim=1)

    with torch.no_grad():
        outputs = model(full_ids, return_dict=True)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    response_length = response_ids.shape[1]
    next_token_logprobs = []
    for i in range(response_length):
        next_token = response_ids[0, i]
        next_token_logprob = log_probs[0, -1 - response_length + i, next_token].item()
        next_token_logprobs.append(next_token_logprob)

    return np.sum(next_token_logprobs)

def main(model_path, proofs_file, output_file):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading proofs...")
    proofs = load_proofs(proofs_file)

    results = []
    for proof in tqdm(proofs):
        logprob = get_proof_logprob(proof, model, tokenizer)
        proof['logprob'] = logprob
        results.append(proof)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--proofs_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    main(args.model_path, args.proofs_file, args.output_file)
