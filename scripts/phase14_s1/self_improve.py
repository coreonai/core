"""Phase 14 S1: 5-seed variance bound on Qwen2.5-Coder-0.5B + LoRA.

Fork of `scripts/phase9_s5/self_improve.py` with:
  1. CHALLENGES imported from `phase14_s1/problems.py` (25 problems
     instead of S5's 11).
  2. `--seed` CLI flag controlling both torch RNG (LoRA init
     randomness) and the seed_base for sampling.
  3. Output JSON named with the seed for easy multi-run aggregation.

Usage:
    CUDA_VISIBLE_DEVICES=0 /tmp/p14_env/bin/python self_improve.py \\
        --seed 0 --rounds 3 --samples 8 --train-steps 60

Run 5 seeds → 5 JSONs → analyze.py for variance bound.
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

# Local import of the 25-problem set
sys.path.insert(0, str(Path(__file__).parent))
from problems import CHALLENGES  # noqa: E402


def verify(prompt, completion, suffix, timeout=2.0):
    program = prompt + completion + suffix
    try:
        out = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, timeout=timeout, text=True,
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
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        return_dict_in_generate=True, output_scores=True,
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


def harvest_round(model, tokenizer, samples_per_prompt, seed_base,
                  max_new_tokens, temperature, top_p):
    records = []
    for ci, ch in enumerate(CHALLENGES):
        for j in range(samples_per_prompt):
            comp, sum_lp, mean_lp, n_tok = generate_and_score(
                model, tokenizer, ch["prompt"],
                max_new_tokens, temperature, top_p,
                seed_base + ci * 10000 + j,
            )
            if n_tok == 0:
                continue
            verdict = verify(ch["prompt"], comp, ch["suffix"])
            records.append({
                "challenge": ch["name"], "completion": comp,
                "n_tokens": n_tok, "sum_logp": sum_lp,
                "mean_logp": mean_lp, "verdict": verdict,
            })
    return records


class CompletionDataset(Dataset):
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
        labels[: prompt_ids.shape[0]] = -100
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


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
    # Phase 14 S1: seed for variance measurement
    ap.add_argument("--seed", type=int, default=0,
                    help="Master seed: controls torch global RNG (LoRA init) "
                         "and seed_base offset for harvest sampling")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to "
                         "scripts/phase14_s1/run_seed{N}.json")
    args = ap.parse_args()

    # Set torch global RNG so LoRA init is reproducible per seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or f"/raid/users/paul/workLLM/scripts/phase14_s1/run_seed{args.seed}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P14S1] seed={args.seed} device={device} model={args.model}")
    print(f"[P14S1] {len(CHALLENGES)} challenges from problems.py")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P14S1] base loaded in {time.time() - t0:.1f}s; "
          f"params={sum(p.numel() for p in base.parameters()) / 1e6:.1f}M")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    history = []
    seed_base = args.seed * 1_000_000  # large stride so seeds don't overlap
    for r in range(args.rounds + 1):
        is_post_train = (r == args.rounds)
        label = f"round-{r}" if not is_post_train else f"final-{r}"
        print(f"\n========== seed={args.seed} {label} ==========")
        records = harvest_round(
            model, tokenizer, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        seed_base += 100 * len(CHALLENGES)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        pass_rate = n_pass / n if n else 0.0
        per_ch = defaultdict(lambda: [0, 0])
        for rec in records:
            per_ch[rec["challenge"]][1] += 1
            if rec["verdict"]:
                per_ch[rec["challenge"]][0] += 1
        print(f"  total={n} pass={n_pass} ({pass_rate:.3f})")
        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": pass_rate,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })

        if is_post_train:
            break

        pairs = [(r_["challenge"], r_["completion"]) for r_ in records if r_["verdict"]]
        # Re-attach prompts (challenge name -> prompt)
        ch_prompt = {ch["name"]: ch["prompt"] for ch in CHALLENGES}
        pairs = [(ch_prompt[ch_name], comp) for ch_name, comp in pairs]
        if len(pairs) < 2:
            print("  [WARN] too few verifier-passed; skipping LoRA step")
            continue
        t = time.time()
        loss = lora_finetune(model, tokenizer, pairs,
                             steps=args.train_steps, batch_size=args.batch_size,
                             lr=args.lr, device=device)
        print(f"  LoRA: {len(pairs)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P14S1 seed={args.seed}] wrote {out_path}")
    print(f"\n=== summary: pass rate per round (seed={args.seed}) ===")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
