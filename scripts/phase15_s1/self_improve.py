"""Phase 15 S1 — substrate variance bound at HumanEval (164 problems).

Phase 14 substrate (25 single-line problems) hit 84% saturation under
SFT. Phase 15 moves to HumanEval full to give algorithmic comparisons
real headroom — target SFT mean 50-70%, σ ≤ 0.03.

Differences from Phase 14 self_improve:
- 164 multi-line problems (HumanEval canonical)
- max_new_tokens 256 (multi-line completions)
- Don't truncate at first \\n — instead truncate at the SECOND blank
  line OR top-level def/class (function-body boundary heuristic)
- Verifier runs the test suite via subprocess
- Generation seed scheme adjusted for 164 problems (was 25)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from problems import CHALLENGES  # noqa: E402


def truncate_completion(text):
    """HumanEval completions need to end at the function body's natural
    boundary. Truncate at the first top-level def/class/import after
    the function body, OR at the first blank line followed by
    non-indented text."""
    # Strip a leading newline if model emits one
    lines = text.split("\n")
    out = []
    for line in lines:
        # Top-level def/class/import after we have some body content
        if out and re.match(r"^(def |class |import |from |if __name__|print\()", line):
            break
        out.append(line)
    # Trim trailing whitespace lines
    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out)


def verify(prompt, completion, suffix, timeout=4.0):
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
def generate_completion(model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = out[0, prompt_len:].tolist()
    if not completion_ids:
        return "", 0
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    text = truncate_completion(text)
    return text, len(completion_ids)


def harvest_round(model, tokenizer, challenges, samples_per_prompt, seed_base,
                  max_new_tokens, temperature, top_p, verify_timeout):
    records = []
    for ci, ch in enumerate(challenges):
        for j in range(samples_per_prompt):
            comp, n_tok = generate_completion(
                model, tokenizer, ch["prompt"],
                max_new_tokens, temperature, top_p,
                seed_base + ci * 10000 + j,
            )
            if n_tok == 0:
                continue
            verdict = verify(ch["prompt"], comp, ch["suffix"], timeout=verify_timeout)
            records.append({
                "challenge": ch["name"], "completion": comp,
                "n_tokens": n_tok, "verdict": verdict,
            })
    return records


class SftDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
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
        prompt_ids = self.tok(prompt, return_tensors="pt", truncation=True,
                              max_length=self.max_len).input_ids[0]
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
    ds = SftDataset(pairs, tokenizer)
    pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, steps // 10), num_training_steps=steps,
    )
    model.train()
    step = 0
    last = 0.0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            sched.step()
            last = out.loss.item()
            step += 1
    model.eval()
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="for smoke testing — only use first N challenges")
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    challenges = CHALLENGES if args.limit is None else CHALLENGES[: args.limit]

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase15_s1/run_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P15S1] seed={args.seed} device={device} challenges={len(challenges)}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P15S1] base loaded in {time.time() - t0:.1f}s")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    history = []
    seed_base = args.seed * 1_000_000
    for r in range(args.rounds + 1):
        is_post = r == args.rounds
        label = f"round-{r}" if not is_post else f"final-{r}"
        t_round = time.time()
        print(f"\n========== seed={args.seed} {label} ==========")
        records = harvest_round(
            model, tokenizer, challenges, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
            args.verify_timeout,
        )
        seed_base += 100 * len(challenges)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        rate = n_pass / n if n else 0.0
        per_ch = defaultdict(lambda: [0, 0])
        for rec in records:
            per_ch[rec["challenge"]][1] += 1
            if rec["verdict"]:
                per_ch[rec["challenge"]][0] += 1
        print(f"  total={n} pass={n_pass} ({rate:.3f})  {time.time() - t_round:.1f}s")
        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": rate,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })
        if is_post:
            break
        ch_prompt = {ch["name"]: ch["prompt"] for ch in challenges}
        pairs = [(ch_prompt[r_["challenge"]], r_["completion"])
                 for r_ in records if r_["verdict"]]
        if len(pairs) < 2:
            print("  [WARN] too few verifier-passed; skipping LoRA step")
            continue
        t = time.time()
        loss = lora_finetune(model, tokenizer, pairs,
                             steps=args.train_steps, batch_size=args.batch_size,
                             lr=args.lr, device=device)
        print(f"  LoRA-FT: {len(pairs)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed, "n_challenges": len(challenges),
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P15S1 seed={args.seed}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
