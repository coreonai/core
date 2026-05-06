"""Phase 9 S4: harvest (prompt, completion, verdict, mean_logp, sum_logp) from
an external HF model on PythonCodeDomain-style challenges.

Output: /tmp/s4_harvest.json — read by a Rust analyzer to compute AUC + selection
sweep on the same shape as critic_baseline_python.rs."""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Mirrors PythonCodeDomain DEFAULT_PYTHON_CHALLENGES (llm-actors/src/domain/python_code.rs).
CHALLENGES = [
    {
        "name": "equals_5",
        "prompt": "def f():\n    return ",
        "suffix": "\n\nassert f() == 5\n",
    },
    {
        "name": "equals_14_via_doubling",
        "prompt": "def f():\n    return 2 * (",
        "suffix": ")\n\nassert f() == 14\n",
    },
    {
        "name": "len_5_string",
        "prompt": "s = ",
        "suffix": "\n\nassert isinstance(s, str) and len(s) == 5\n",
    },
    # Easier additions: arithmetic single-token completions a 0.5B base
    # can plausibly hit.
    {
        "name": "two_plus_to_5",
        "prompt": "x = 2 + ",
        "suffix": "\nassert x == 5\n",
    },
    {
        "name": "ten_minus_to_3",
        "prompt": "x = 10 - ",
        "suffix": "\nassert x == 3\n",
    },
    {
        "name": "two_pow_to_8",
        "prompt": "x = 2 ** ",
        "suffix": "\nassert x == 8\n",
    },
]


def verify(prompt: str, completion: str, suffix: str, timeout: float = 2.0) -> bool:
    program = prompt + completion + suffix
    try:
        out = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False
    return out.returncode == 0


@torch.no_grad()
def generate_and_score(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
):
    """Generate one completion, return (completion_text, sum_logp, mean_logp, n_tokens)."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_ids = out.sequences[0]
    completion_ids = full_ids[prompt_len:].tolist()
    if not completion_ids:
        return "", 0.0, 0.0, 0

    # Truncate at first \n to match internal Generator's line-completion behavior.
    # Score covers the kept tokens only — log-prob compares apples-to-apples
    # with what the verifier actually saw.
    truncated_ids = []
    truncated_scores = []
    for step_logits, tok_id in zip(out.scores, completion_ids):
        truncated_ids.append(tok_id)
        truncated_scores.append(step_logits)
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        if "\n" in decoded:
            break

    completion_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    # Strip newline for verifier (suffix supplies its own \n).
    if completion_text.endswith("\n"):
        completion_text = completion_text.rstrip("\n")

    sum_logp = 0.0
    n = 0
    for step_logits, tok_id in zip(truncated_scores, truncated_ids):
        log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
        sum_logp += log_probs[tok_id].item()
        n += 1
    mean_logp = sum_logp / max(n, 1)
    return completion_text, sum_logp, mean_logp, n


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    samples_per_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/s4_harvest.json"

    print(f"[S4] loading {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[S4] loaded in {time.time() - t0:.1f}s; n_params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    results = []
    seed_base = 0
    for ch in CHALLENGES:
        print(f"\n[S4] challenge: {ch['name']}")
        n_pass = 0
        for i in range(samples_per_prompt):
            comp, sum_lp, mean_lp, n_tok = generate_and_score(
                model,
                tokenizer,
                ch["prompt"],
                max_new_tokens=24,
                temperature=0.8,
                top_p=0.95,
                seed=seed_base + i,
            )
            if n_tok == 0:
                continue
            verdict = verify(ch["prompt"], comp, ch["suffix"])
            if verdict:
                n_pass += 1
            results.append({
                "challenge": ch["name"],
                "prompt": ch["prompt"],
                "completion": comp,
                "n_tokens": n_tok,
                "sum_logp": sum_lp,
                "mean_logp": mean_lp,
                "verdict": verdict,
            })
        seed_base += samples_per_prompt
        print(f"  pass {n_pass}/{samples_per_prompt}")

    Path(out_path).write_text(json.dumps({
        "model": model_name,
        "samples_per_prompt": samples_per_prompt,
        "challenges": [c["name"] for c in CHALLENGES],
        "samples": results,
    }, indent=2))
    print(f"\n[S4] wrote {out_path} ({len(results)} samples)")


if __name__ == "__main__":
    main()
