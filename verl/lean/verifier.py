from typing import List

import sys
import re
sys.path.append("/home/awhe/projects/DeepSeek-Prover-V1.5")
from prover.lean.verifier import Lean4ServerScheduler

prompt = r'''Complete the following Lean 4 code:

```lean4
'''

code_prefix = r'''import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

'''

def verify_with_deepseek_verifier(
        full_proofs: List[str],
        theorem_statements: List[str], 
        contexts: List[str], 
        max_workers: int = 16,
        penalize_extra_text: bool = False,
    ):

    assert len(full_proofs) % len(theorem_statements) == 0
    assert len(theorem_statements) == len(contexts)

    num_problems = len(theorem_statements)
    num_samples_per_problem = len(full_proofs) // num_problems

    # first we assemble the proof code files
    # which consists of the context, statement, and proof
    # we will use the context hardcoded in the template
    code_files = []
    after_code_texts = []  # New list to store text after the code blocks
    num_parse_fails = 0
    for i in range(len(full_proofs)):
        theorem_statement = theorem_statements[i // num_samples_per_problem]
        proof_body = full_proofs[i]
        code_file = prompt + code_prefix + theorem_statement + proof_body
        # try to parse the code file
        try:
            # Extract the code part
            code_match = re.search(r'```lean4\n(.*?)\n```', code_file, re.DOTALL)
            if code_match:
                code_file_parsed = code_match.group(1)
                code_files.append(code_file_parsed)
                
                # Extract everything after the closing ```
                end_pos = code_match.end()
                after_code_text = code_file[end_pos:].strip()
                after_code_texts.append(after_code_text)
            else:
                raise Exception("No code block found")
        except Exception as e:
            # this should result in a verification failure
            num_parse_fails += 1
            code_files.append("failed to parse")
            after_code_texts.append("")  # Add empty string for failed parses

    print(f"[VERIFIER] {num_parse_fails} parse failures")
    # print(code_files[0])
    # print(code_files[-1])

    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=max_workers, 
        timeout=300, 
        memory_limit=10, 
        name='verifier'
    )

    request_id_list = lean4_scheduler.submit_all_request(code_files)
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()

    # print(outputs_list[0])
    # print(outputs_list[-1])

    final_outputs = []
    for i in range(num_problems):
        outputs = outputs_list[i * num_samples_per_problem:(i + 1) * num_samples_per_problem]
        correct = [output["complete"] for output in outputs]

        if penalize_extra_text:
            # extra_text = after_code_texts[i * num_samples_per_problem:(i + 1) * num_samples_per_problem]
            # has_extra_text = [len(text.strip()) > 0 for text in extra_text]
            text = full_proofs[i * num_samples_per_problem:(i + 1) * num_samples_per_problem]
            has_extra_text = [not t.strip().endswith("```") for t in text]
            for j in range(num_samples_per_problem):
                if has_extra_text[j]:
                    correct[j] = False
                    print(f"[VERIFIER] Penalized extra text for sample {j}")
                    print(text[j])
        
        success_indices = [i for i, s in enumerate(correct) if s]
        incomplete = [not output["complete"] for output in outputs]
        final_outputs.append(dict(
            success = any(correct),
            num_success = sum(correct),
            success_indices = success_indices,
            num_errors = sum(incomplete),
            msg = "Search successful" if any(correct) else f"Search ended with {sum(incomplete)} errors",
        ))
    
    return final_outputs
