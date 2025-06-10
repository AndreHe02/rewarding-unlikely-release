import os
import glob
import jsonlines
import argparse
import pandas as pd
from verl.lean.utils import compute_pass_metrics, compute_response_metrics


def load_proofs_and_compute_metrics(dataset_path, proofs_path, num_samples):
    dataset = pd.read_parquet(dataset_path)
    proofs = []
    file_pattern = os.path.join(proofs_path, f"proofs_device*.jsonl")
    # filter out batch files, which would double-count proofs
    file_pattern = [f for f in glob.glob(file_pattern) if "batch" not in f]
    for file in file_pattern:
        with jsonlines.open(file, mode='r') as reader:
            for item in reader:
                proofs.append(item)

    metrics = {}
    pass_metrics = compute_pass_metrics(dataset, proofs, num_samples=num_samples)
    metrics.update(pass_metrics)

    response_metrics = compute_response_metrics(dataset, proofs)
    metrics.update(response_metrics)
    
    return metrics

def load_proofs(proofs_path):
    proofs = []
    file_pattern = os.path.join(proofs_path, f"proofs_device*.jsonl")
    # filter out batch files, which would double-count proofs
    file_pattern = [f for f in glob.glob(file_pattern) if "batch" not in f]
    for file in file_pattern:
        with jsonlines.open(file, mode='r') as reader:
            for item in reader:
                proofs.append(item)
    return proofs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check progress of theorem proving')
    parser.add_argument('--name', type=str, default='base-eval',
                        help='Name of the evaluation')
    parser.add_argument('--dataset', type=str, default="minif2f-test")
    parser.add_argument('--num_samples', type=int, default=128)
    args = parser.parse_args()

    dataset_dict = {
        "minif2f-test": "/home/awhe/projects/verl/data/minif2f_test.parquet",
        "unseen": "/home/awhe/projects/verl/data/mff-lwb-10k-unseen.parquet",
    }

    proofs_path = f'/data/user_data/awhe/model-evals/{args.name}/'
    metrics = load_proofs_and_compute_metrics(dataset_dict[args.dataset], proofs_path, args.num_samples)
    print(metrics)

    # TODO length, truncation, parsing success rates might also be useful
