# prompts for internlm2-step-prover
policy_prompt_template_step_prover = """<|im_start|>user
DECL {0}
GOAL {1}
<|im_end|>
<|im_start|>assistant
PROOFSTEP """

# prompts for mctx model
policy_prompt_template_mctx = """/- You are proving a theorem in Lean 4.
You are given the following information:
- The current proof state, inside [STATE]...[/STATE]

Your task is to generate the next tactic in the proof.
Put the next tactic inside [TAC]...[/TAC]
-/
[STATE]
{1}
[/STATE]
[TAC]
"""

# prompts for deepseek prover
policy_prompt_template_deepseek = r'''Complete the following Lean 4 code:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

'''

policy_prompt_template_deepseek_cot = r'''Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

'''


def prompt_fn_step_prover(formal_statement, informal_statement, theorem_full_name, goal, history=None):
    return policy_prompt_template_step_prover.format(theorem_full_name, goal)

def prompt_fn_mctx(formal_statement, informal_statement, theorem_full_name, goal, history=None):
    return policy_prompt_template_mctx.format(theorem_full_name, goal)

def prompt_fn_deepseek(formal_statement, informal_statement, theorem_full_name, goal, history=None):
    prompt = policy_prompt_template_deepseek + informal_statement + formal_statement + '<｜begin▁of▁sentence｜>'
    if history is not None:
        prompt += history
    return prompt

def prompt_fn_deepseek_cot(formal_statement, informal_statement, theorem_full_name, goal, history=None):
    prompt = policy_prompt_template_deepseek_cot + informal_statement + formal_statement + '<｜begin▁of▁sentence｜>'
    if history is not None:
        prompt += history
    return prompt

prompt_fns_by_model = {
    "internlm2-step-prover": prompt_fn_step_prover,
    "ntp-mathlib-context-deepseek-coder-1.3b": prompt_fn_mctx,
    "deepseek-prover": prompt_fn_deepseek,
    "deepseek-prover-cot": prompt_fn_deepseek_cot,
}

def get_prompt_fn(model_name):
    return prompt_fns_by_model[model_name]

terminal_tokens = {
    "deepseek-prover": "```",
    "deepseek-prover-cot": "```",
}

def get_terminal_token(model_name):
    if model_name in terminal_tokens:
        return terminal_tokens[model_name]
    else:
        return None


def response_fn_deepseek(proof_body, terminal):
    if terminal:
        if proof_body.endswith("\n"):
            return proof_body + "```\n"
        else:
            return proof_body + "\n```\n"
    else:
        return proof_body
    
response_fns_by_model = {
    "deepseek-prover": response_fn_deepseek,
    "deepseek-prover-cot": response_fn_deepseek,
}

def get_response_fn(model_name):
    if model_name in response_fns_by_model:
        return response_fns_by_model[model_name]
    else:
        return lambda proof_body, terminal: proof_body
