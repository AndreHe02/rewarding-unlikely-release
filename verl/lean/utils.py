
def get_goal(output):
    if "sorries" in output and len(output["sorries"]) > 0:
        if "goals" in output["sorries"][0] and len(output["sorries"][0]["goals"]) > 0:
            return output["sorries"][0]["goals"][0]
        if "goal" in output["sorries"][0]:
            return output["sorries"][0]["goal"]
    if "goals" in output and len(output["goals"]) > 0:
        return output["goals"][0]
    return None


def get_goals(output):
    if "sorries" in output and len(output["sorries"]) > 0:
        if "goals" in output["sorries"][0] and len(output["sorries"][0]["goals"]) > 0:
            return output["sorries"][0]["goals"][0]
        if "goal" in output["sorries"][0]:
            return output["sorries"][0]["goal"]
    if "goals" in output and len(output["goals"]) > 0:
        return [goal for goal in output["goals"]]
    return None


def parse_tactic(output, stop_tokens=["<|im_end|>", "[/TAC]"]):
    # TODO remove comments first
    # We aren't working with any CoT step provers now
    # so this is not needed
    for stop_token in stop_tokens:
        if stop_token in output:
            output = output[:output.index(stop_token)]
            break
    return output.strip()


def concat_tactics(tactics):
    return "".join([t if t.endswith("\n") else t + "\n" for t in tactics])

def terminate_proof(body, terminate_token):
    if body.endswith("\n"):
        return body + terminate_token
    else:
        return body + "\n" + terminate_token

def _error_message(output):
    if output == None:
        return True
    if "messages" in output:
        for d in output["messages"]:
            return True
    if "message" in output:
        if "error" in output["message"]:
            return True
    return False


def _parse_output(output, proof_state, next_tactic):
    if not _error_message(output):
        if output == "" and "sorry" not in output:
            return {"status": "invalid"}
        # elif example["full_name"] != None and example["full_name"] in next_tactic:
        #     # forbid recursion
        #     return {"status": "invalid"}
        elif output["goals"] == [] and len(output) == 2:
            return {"status": "done", "output": output}
        elif output["proofState"] > proof_state:
            return {"status": "valid", "output": output}
        else:
            return {"status": "invalid"}
    return {"status": "invalid"}


def eval_tactic(next_tactic, thread, proof_state):
    if next_tactic is None:
        return {"status": "invalid", "reason": "no tactic provided"}
    
    if "admit" in next_tactic or "sorry" in next_tactic:
        return {"status": "invalid"}

    output = thread.submit_and_receive(
        {"tactic": next_tactic, "proofState": proof_state}
    )

    if output is None or output.get("error") == "timeout":
        return {"status": "invalid", "reason": "timeout"}

    return _parse_output(output, proof_state, next_tactic)


def combine_multiline_comments(lines: list) -> list:
    """
    Processes a list of lines and combines multi-line comment blocks into single logical lines.
    A multi-line comment block starts with a line that (after stripping) begins with '/-'
    and continues until a line that contains '-/' is found.
    """
    combined_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        # If the line starts a multi-line comment block.
        if stripped.startswith("/-"):
            comment_block = [line]
            i += 1
            # Continue appending lines until we find the closing marker.
            while i < n:
                comment_line = lines[i]
                comment_block.append(comment_line)
                if "-/" in comment_line:
                    i += 1
                    break
                i += 1
            # Combine the block into a single string separated by newlines.
            combined_lines.append("\n".join(comment_block))
        else:
            combined_lines.append(line)
            i += 1
    return combined_lines

def is_comment_line(logical_line: str) -> bool:
    """
    Returns True if the logical line is (likely) a comment.
    This applies if the line (or block) starts with '--' or '/-'.
    """
    stripped = logical_line.lstrip()
    return stripped.startswith("--") or stripped.startswith("/-")

def parse_proof_steps(proof_body: str, retain_comments: bool = False, terminal_token: str = None) -> list:
    """
    Parses a Lean4 proof body into individual steps while preserving the absolute indentation
    of each line and retaining inline/indented comments. In this version, multi-line comment blocks 
    (delimited by '/-' and '-/') are combined and treated as a single comment entry.
    
    Process:
      1. Split the proof body into lines.
      2. Combine multi-line comment blocks into single logical lines.
      3. Determine the minimal indentation among non-blank, non-comment logical lines.
      4. Group lines into steps:
         - A new step begins with a non-comment logical line having minimal indentation.
         - Indented lines (or logical lines) are considered part of the current step.
         - Top-level comment logical lines (i.e. with minimal indentation) immediately preceding a step
           are buffered and attached to that step if `retain_comments` is True.
    
    Returns:
      A list of strings, each representing one parsed proof step with its original indentation preserved.
    """

    if terminal_token is not None:
        end_idx = proof_body.find(terminal_token)
        if end_idx != -1:
            proof_body = proof_body[:end_idx]

    # Split the proof body into raw lines.
    raw_lines = proof_body.splitlines()
    # Combine multi-line comment blocks.
    logical_lines = combine_multiline_comments(raw_lines)
    
    # Determine minimal indent from non-blank, non-comment lines.
    min_indent = None
    for line in logical_lines:
        if line.strip() and not is_comment_line(line):
            indent = len(line) - len(line.lstrip())
            if min_indent is None or indent < min_indent:
                min_indent = indent
    if min_indent is None:
        min_indent = 0  # fallback if no code lines are found
    
    steps = []
    current_step_lines = []
    # Buffer for top-level comment logical lines that might be attached.
    comment_buffer = []
    
    for logical_line in logical_lines:
        if not logical_line.strip():
            # Blank line: clear any pending comment buffer.
            comment_buffer = []
            continue
        
        # Determine the indentation from the first line of the logical block.
        first_line = logical_line.splitlines()[0]
        current_indent = len(first_line) - len(first_line.lstrip())
        
        if is_comment_line(logical_line) and current_indent <= min_indent:
            # This is a top-level comment logical block.
            if retain_comments:
                comment_buffer.append(logical_line)
            # When not retaining, simply ignore it.
            continue
        
        # For a non-comment (or indented comment block) logical line.
        if current_indent == min_indent:
            # Start a new step.
            if current_step_lines:
                steps.append("\n".join(current_step_lines))
                current_step_lines = []
            if retain_comments and comment_buffer:
                current_step_lines.extend(comment_buffer)
            comment_buffer = []
            current_step_lines.append(logical_line)
        else:
            # This logical line is indented relative to the top-level.
            current_step_lines.append(logical_line)
            # Clear any pending comment buffer.
            comment_buffer = []
    
    if current_step_lines:
        steps.append("\n".join(current_step_lines))
    
    return steps


def remove_comments(text: str) -> str:
    """
    Removes all Lean-style comments from the input text.
    This includes:
      - Single-line comments starting with '--'
      - Nested multi-line block comments delimited by '/-' and '-/'
    
    Returns:
      The input text with all comments removed.
    """
    i = 0
    n = len(text)
    result = []
    
    while i < n:
        # Check for single-line comment.
        if text[i:i+2] == '--':
            newline_index = text.find('\n', i)
            if newline_index == -1:
                break  # Reached end of text.
            else:
                i = newline_index
        # Check for block comment.
        elif text[i:i+2] == '/-':
            i += 2  # Skip the '/-' marker.
            depth = 1
            while i < n and depth > 0:
                # Start of nested block comment.
                if text[i:i+2] == '/-':
                    depth += 1
                    i += 2
                # End of a block comment.
                elif text[i:i+2] == '-/':
                    depth -= 1
                    i += 2
                else:
                    i += 1
        else:
            result.append(text[i])
            i += 1
    
    # remove empty lines
    result = ''.join(result)
    result_lines = result.splitlines()
    result_lines = [line for line in result_lines if line.strip()]
    return '\n'.join(result_lines)

# Example usage:
# if __name__ == "__main__":
#     sample_proof = """
#     -- Top-level comment for the first tactic
#     intro h  -- inline comment for intro
#       -- indented comment within the step
#     /- 
#       Multi-line comment for the have block:
#       explaining the tactic in detail
#     -/
#     have p : Prop := 
#       /- Block comment explaining the tactic -/
#       sorry
#     exact p  -- final tactic with inline comment
#     -- test
#     -- another test
#     """

#     sample_proof = """
#     have h : e = 7 - x := by
#         linear_combination h₀
#     rw [h] at h₁
# -- Porting note: added the next line
#     have : x = -4 := by
#         linear_combination h₂
#     simp_all
# -- Porting note: added the next line
# -- linear_combination h₀ - this
# -- Porting note: the first `linear_combination` was `linear_combination h₀`
#      <;> test
#      <;> fake indent 
#     """
#     print("Steps with comments retained:")
#     steps_with_comments = parse_proof_steps(sample_proof, retain_comments=True)
#     for idx, step in enumerate(steps_with_comments, start=1):
#         print(f"Step {idx}:\n{step}\n{'-'*40}")
    
#     print("\nSteps without top-level comments (but inline/indented remain):")
#     steps_without_comments = parse_proof_steps(sample_proof, retain_comments=False)
#     for idx, step in enumerate(steps_without_comments, start=1):
#         print(f"Step {idx}:\n{step}\n{'-'*40}")

# # Example usage:
# if __name__ == "__main__":
#     sample_proof = '''  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,
#     Nat.succ_add]
# /-- this is a comment -/
#   have h₁' : a * r = 2 := by simpa [h₀] using h₁
#   have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂
#   have h₃ : r ^ 2 = 3 := by
#     nlinarith
#   have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
#     apply eq_or_eq_neg_of_sq_eq_sq <;>
#     field_simp <;>
#     nlinarith
#   simpa [h₀] using h₄
# ```
# '''
#     parsed_steps = parse_proof_steps(sample_proof)
#     for idx, step in enumerate(parsed_steps, start=1):
#         print(f"Step {idx}:\n{step}\n{'-'*40}")

def make_lean_proof(
        theorem_name: str,
        theorem_statement: str,
        informal_statement: str,
        proof: str,
        correct: bool = None,
        context: str = None,
):
    return {
        "theorem_name": theorem_name,
        "theorem_statement": theorem_statement,
        "informal_statement": informal_statement,   
        "proof": proof,
        **({"context": context} if context is not None else {}),
        **({"correct": correct} if correct is not None else {}),
    }

import math
def compute_pass_at_n(s, a, n):
    # s: number of successes
    # a: number of attempts
    # n: pass at n
    f = a - s
    if f < n:
        return 1.0
    else:
        return 1 - math.comb(f, n) / math.comb(a, n)
    

def uses_sorry(proof: str) -> bool:
    proof_steps = parse_proof_steps(proof, retain_comments=False, terminal_token="```")
    for step in proof_steps:
        if "sorry" in step:
            return True
    return False


from collections import defaultdict
def compute_pass_metrics(dataset, proofs, num_samples, prefix: str = "train"):
    theorem_full_names = list(dataset["theorem_full_name"].unique())
    problem_successes = defaultdict(int)
    problem_attempts = defaultdict(int)

    for proof in proofs:
        theorem_full_name = proof["theorem_name"]
        success = proof["correct"] if "correct" in proof else False
        # TODO this should be handled by the verifier
        if uses_sorry(proof["proof"]):
            success = False
        problem_successes[theorem_full_name] += success
        problem_attempts[theorem_full_name] += 1

    num_insufficient_attempts = 0
    num_excessive_attempts = 0
    for theorem_full_name in theorem_full_names:
        if problem_attempts[theorem_full_name] < num_samples:
            num_insufficient_attempts += 1
            # print(f"Warning: {theorem_full_name} has fewer attempts than num_samples: {problem_attempts[theorem_full_name]}")
        if problem_attempts[theorem_full_name] > num_samples:
            num_excessive_attempts += 1
            # print(f"Warning: {theorem_full_name} has more attempts than num_samples: {problem_attempts[theorem_full_name]}")

    if num_insufficient_attempts > 0:
        print(f"Warning: {num_insufficient_attempts} theorems have fewer attempts than num_samples: {num_samples}")
    if num_excessive_attempts > 0:
        print(f"Warning: {num_excessive_attempts} theorems have more attempts than num_samples: {num_samples}")

    pass_at_n = defaultdict(int)
    n = 1
    while n <= num_samples:
        for theorem_full_name in theorem_full_names:
            if theorem_full_name not in problem_successes:
                continue
            if problem_attempts[theorem_full_name] < n:
                continue
            prob_pass_at_n = compute_pass_at_n(problem_successes[theorem_full_name], problem_attempts[theorem_full_name], n)
            pass_at_n[n] += prob_pass_at_n
        n *= 2

    pass_at_n = {n: pass_at_n[n] / len(theorem_full_names) for n in pass_at_n}

    final_metrics = {
        f"{prefix}/pass_at_{n}": pass_at_n[n] for n in pass_at_n
    }

    return final_metrics

def compute_response_metrics(dataset, proofs, prefix: str = "train", terminal_token: str = "```"):
    theorem_full_names = list(dataset["theorem_full_name"].unique())
    total_unique_proofs = 0
    total_finished_proofs = 0

    proofs_by_theorem = defaultdict(list)
    for proof in proofs:
        theorem_full_name = proof["theorem_name"]
        proofs_by_theorem[theorem_full_name].append(proof["proof"])

    for problem_name in proofs_by_theorem:
        problem_proofs = proofs_by_theorem[problem_name]
        unique_proofs = set()
        
        for proof in problem_proofs:
            proof_key = parse_proof_steps(proof, retain_comments=False, terminal_token=terminal_token)
            proof_key = "\n".join(proof_key)
            unique_proofs.add(proof_key)

            if terminal_token in proof:
                total_finished_proofs += 1

        num_unique_proofs = len(unique_proofs)
        total_unique_proofs += num_unique_proofs

    final_metrics = {
        f"{prefix}/total_unique_proofs": total_unique_proofs,
        f"{prefix}/unique_ratio": total_unique_proofs / len(proofs),
        # f"{prefix}/total_finished_proofs": total_finished_proofs,
        f"{prefix}/finished_ratio": total_finished_proofs / len(proofs),
    }

    return final_metrics