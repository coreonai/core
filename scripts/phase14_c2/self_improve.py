"""Phase 14 C2: Muon vs AdamW for LoRA adapters at Qwen scale.

Fork of phase14_s1/self_improve.py with `--optimizer adam|muon`.
Muon orthogonalizes gradient updates on the LoRA delta_A / delta_B
2-D matrices via Newton-Schulz before applying them.

Question this answers: Phase 12 S1 reported "+78% Muon gen" at K9
1M (retracted by Phase 13 S1 as seed-0 outlier). Does Muon actually
help LoRA training at the quieter Qwen substrate (Phase 14 S1
σ=0.011)?

Threshold: 2σ ≈ 0.022 absolute final-pass-rate delta = robust win.
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase14_s1"))
from muon import Muon  # noqa: E402
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


def lora_finetune(model, tokenizer, pairs, steps, batch_size, lr, device, optimizer_kind):
    if not pairs:
        return 0.0
    ds = CompletionDataset(pairs, tokenizer)
    pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimizer_kind == "muon":
        opt = Muon(trainable, lr=lr, momentum=0.95, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, steps // 10), num_training_steps=steps,
    )
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adam", "muon"], default="adam",
                    help="Phase 14 C2: choose AdamW or Muon for LoRA training")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase14_c2/run_{args.optimizer}_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P14C2] seed={args.seed} optimizer={args.optimizer} device={device}")
    print(f"[P14C2] {len(CHALLENGES)} challenges")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P14C2] base loaded in {time.time() - t0:.1f}s")

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
        print(f"\n========== seed={args.seed} {args.optimizer} {label} ==========")
        records = harvest_round(
            model, tokenizer, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        seed_base += 100 * len(CHALLENGES)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        rate = n_pass / n if n else 0.0
        per_ch = defaultdict(lambda: [0, 0])
        for rec in records:
            per_ch[rec["challenge"]][1] += 1
            if rec["verdict"]:
                per_ch[rec["challenge"]][0] += 1
        print(f"  total={n} pass={n_pass} ({rate:.3f})")
        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": rate,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })
        if is_post:
            break
        ch_prompt = {ch["name"]: ch["prompt"] for ch in CHALLENGES}
        pairs = [(ch_prompt[r_["challenge"]], r_["completion"])
                 for r_ in records if r_["verdict"]]
        if len(pairs) < 2:
            print("  [WARN] too few verifier-passed; skipping LoRA step")
            continue
        t = time.time()
        loss = lora_finetune(model, tokenizer, pairs,
                             steps=args.train_steps, batch_size=args.batch_size,
                             lr=args.lr, device=device, optimizer_kind=args.optimizer)
        print(f"  LoRA[{args.optimizer}]: {len(pairs)} pairs × "
              f"{args.train_steps} steps, last_loss={loss:.3f}, "
              f"{time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed, "optimizer": args.optimizer,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P14C2 seed={args.seed} {args.optimizer}] wrote {out_path}")
    print(f"\n=== summary: pass rate per round (seed={args.seed} {args.optimizer}) ===")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
