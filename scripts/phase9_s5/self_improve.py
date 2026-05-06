"""Phase 9 S5: external-model self-improve loop with LoRA + Shape-C critic.

Builds on Phase 9 S4 (which proved Qwen2.5-Coder-0.5B has sum-AUC 0.702
+ F=8 lift 1.95× on Python challenges) by closing the loop:

  for round in 0..N:
      generate K candidates per challenge
      compute sum log-prob (Shape-C critic, no separate model)
      verify with `python -c`
      track baseline gen pass rate AND critic-rerank pass rate
      build training set: (prompt, completion) for verifier-passed samples
      LoRA fine-tune on training set
      reload adapter
      next round

Two questions this script answers:
1. Does an external 1B-scale model self-improve under the same Shape-C loop
   we use for in-house 1M models?
2. Does the critic lift hold (or grow) as the model improves?

Real-world task application: 6 S4 challenges (slot-fill arithmetic) +
5 HumanEval-style function-completion challenges (single-line returns).

Usage:
    CUDA_VISIBLE_DEVICES=0 python self_improve.py \\
        --model Qwen/Qwen2.5-Coder-0.5B \\
        --rounds 3 --samples 8 --train-steps 60
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Challenge set — S4's 6 + 5 HumanEval-style real-world function completions.
# All targets are single-line completions (truncated at \n) so the
# verifier loop matches the in-house Generator behavior.
# ---------------------------------------------------------------------------

CHALLENGES = [
    # === S4 carried-over (kept for continuity with the S4 measurement) ===
    {"name": "equals_5", "prompt": "def f():\n    return ",
     "suffix": "\n\nassert f() == 5\n"},
    {"name": "equals_14_via_doubling", "prompt": "def f():\n    return 2 * (",
     "suffix": ")\n\nassert f() == 14\n"},
    {"name": "len_5_string", "prompt": "s = ",
     "suffix": "\n\nassert isinstance(s, str) and len(s) == 5\n"},
    {"name": "two_plus_to_5", "prompt": "x = 2 + ", "suffix": "\nassert x == 5\n"},
    {"name": "ten_minus_to_3", "prompt": "x = 10 - ", "suffix": "\nassert x == 3\n"},
    {"name": "two_pow_to_8", "prompt": "x = 2 ** ", "suffix": "\nassert x == 8\n"},

    # === Real-world: HumanEval-style single-line function bodies ===
    {
        "name": "is_even",
        "prompt": 'def is_even(n):\n    """Return True if n is even."""\n    return ',
        "suffix": "\n\nassert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True\n",
    },
    {
        "name": "list_sum",
        "prompt": 'def total(xs):\n    """Return the sum of a list of numbers."""\n    return ',
        "suffix": "\n\nassert total([]) == 0\nassert total([1, 2, 3]) == 6\nassert total([5]) == 5\n",
    },
    {
        "name": "is_positive",
        "prompt": 'def is_positive(n):\n    """Return True if n > 0."""\n    return ',
        "suffix": "\n\nassert is_positive(5) == True\nassert is_positive(0) == False\nassert is_positive(-3) == False\n",
    },
    {
        "name": "count_chars",
        "prompt": 'def count(s, ch):\n    """Return how many times ch appears in s."""\n    return ',
        "suffix": "\n\nassert count('hello', 'l') == 2\nassert count('', 'a') == 0\nassert count('abc', 'b') == 1\n",
    },
    {
        "name": "double_it",
        "prompt": 'def double(x):\n    """Return x doubled."""\n    return ',
        "suffix": "\n\nassert double(3) == 6\nassert double(0) == 0\nassert double(-2) == -4\n",
    },
]


def verify(prompt, completion, suffix, timeout=2.0):
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
def generate_and_score(model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed):
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

    truncated_ids, truncated_scores = [], []
    for step_logits, tok_id in zip(out.scores, completion_ids):
        truncated_ids.append(tok_id)
        truncated_scores.append(step_logits)
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        if "\n" in decoded:
            break

    completion_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
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


def harvest_round(model, tokenizer, samples_per_prompt, seed_base, max_new_tokens, temperature, top_p):
    """Generate K candidates per challenge; return list of records + summary metrics."""
    records = []
    for ci, ch in enumerate(CHALLENGES):
        for j in range(samples_per_prompt):
            comp, sum_lp, mean_lp, n_tok = generate_and_score(
                model, tokenizer, ch["prompt"],
                max_new_tokens, temperature, top_p,
                seed_base + ci * 1000 + j,
            )
            if n_tok == 0:
                continue
            verdict = verify(ch["prompt"], comp, ch["suffix"])
            records.append({
                "challenge": ch["name"], "prompt": ch["prompt"], "suffix": ch["suffix"],
                "completion": comp, "n_tokens": n_tok,
                "sum_logp": sum_lp, "mean_logp": mean_lp, "verdict": verdict,
            })
    return records


def selection_lift(records, k):
    """Return (random_pass, critic_pass) by per-challenge cohort, picking 1 of K."""
    by_ch = defaultdict(list)
    for r in records:
        by_ch[r["challenge"]].append(r)
    rand_pass = 0
    crit_pass = 0
    total = 0
    import random as pyrandom
    rng = pyrandom.Random(0xB00B + k)
    n_trials = 100
    for _ in range(n_trials):
        for cohort in by_ch.values():
            if len(cohort) < k:
                continue
            picks = rng.sample(cohort, k)
            if picks[0]["verdict"]:
                rand_pass += 1
            best = max(picks, key=lambda r: r["sum_logp"])
            if best["verdict"]:
                crit_pass += 1
            total += 1
    if total == 0:
        return 0.0, 0.0
    return rand_pass / total, crit_pass / total


class CompletionDataset(Dataset):
    """Tokenize (prompt + completion + \\n) supervised on completion tokens only."""
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        prompt, completion = self.pairs[i]
        full = prompt + completion + "\n"
        enc = self.tok(full, return_tensors="pt", truncation=True, max_length=self.max_len)
        ids = enc.input_ids[0]
        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        labels = ids.clone()
        labels[: prompt_ids.shape[0]] = -100  # mask prompt
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def lora_finetune(model, tokenizer, pairs, steps, batch_size, lr, device):
    if not pairs:
        return 0.0
    ds = CompletionDataset(pairs, tokenizer)
    pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, steps // 10),
                                            num_training_steps=steps)
    model.train()
    step = 0
    last_loss = 0.0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            last_loss = loss.item()
            step += 1
    model.eval()
    return last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--out", default="/raid/users/paul/workLLM/scripts/phase9_s5/run.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S5] device={device} model={args.model}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[S5] base loaded in {time.time() - t0:.1f}s; params={sum(p.numel() for p in base.parameters()) / 1e6:.1f}M")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    history = []
    seed_base = 0
    for r in range(args.rounds + 1):  # +1 for the post-training eval round
        is_post_train = (r == args.rounds)
        label = f"round-{r}" if not is_post_train else f"final-{r}"
        print(f"\n========== {label} ==========")
        records = harvest_round(
            model, tokenizer, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        seed_base += 100 * len(CHALLENGES)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        pass_rate = n_pass / n if n else 0.0
        rp2, cp2 = selection_lift(records, 2)
        rp4, cp4 = selection_lift(records, 4)
        rp8, cp8 = selection_lift(records, 8)
        lift4 = (cp4 / rp4) if rp4 > 1e-6 else float("inf")
        lift8 = (cp8 / rp8) if rp8 > 1e-6 else float("inf")
        print(f"  total={n} pass={n_pass} ({pass_rate:.3f})")
        print(f"  F=2  rand={rp2:.3f} crit={cp2:.3f}")
        print(f"  F=4  rand={rp4:.3f} crit={cp4:.3f}  lift={lift4:.2f}x")
        print(f"  F=8  rand={rp8:.3f} crit={cp8:.3f}  lift={lift8:.2f}x")

        per_ch = defaultdict(lambda: [0, 0])
        for rec in records:
            per_ch[rec["challenge"]][1] += 1
            if rec["verdict"]:
                per_ch[rec["challenge"]][0] += 1

        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": pass_rate,
            "F2_rand": rp2, "F2_crit": cp2,
            "F4_rand": rp4, "F4_crit": cp4, "F4_lift": lift4,
            "F8_rand": rp8, "F8_crit": cp8, "F8_lift": lift8,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })

        if is_post_train:
            break

        # Build training set: verifier-passed completions only (clean labels).
        # In a deployment without verifier, we'd use critic-top-K instead.
        pairs = [(rec["prompt"], rec["completion"]) for rec in records if rec["verdict"]]
        print(f"  training_set_size = {len(pairs)}")
        if len(pairs) < 2:
            print("  [WARN] too few verifier-passed; skipping LoRA step")
            continue
        t = time.time()
        loss = lora_finetune(
            model, tokenizer, pairs,
            steps=args.train_steps, batch_size=args.batch_size,
            lr=args.lr, device=device,
        )
        print(f"  LoRA: {len(pairs)} pairs × {args.train_steps} steps, last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[S5] wrote {args.out}")

    print("\n=== summary: pass rate per round ===")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})  "
              f"F=4 lift={h['F4_lift']:.2f}x")


if __name__ == "__main__":
    main()
