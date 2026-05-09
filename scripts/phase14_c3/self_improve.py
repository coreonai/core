"""Phase 14 C3: DPO variants for LoRA at Qwen substrate.

Re-tests Phase 11 S5's K9-1M DPO claims at the quiet Phase 14 substrate
(σ_AdamW = 0.011). Two variants vs SFT (= Phase 14 S1 baseline):

  hybrid    : (1-α)·DPO_loss + α·SFT_chosen,  α=0.3, β=0.1   — Phase 11 S5 best peak
  round0    : pure DPO at round 0, SFT for rounds 1+, β=0.1   — Phase 11 S5 fastest

Reference model: PEFT's `model.disable_adapter()` ctx → frozen base.
At LoRA init, delta_B=0 → policy ≡ ref, so DPO is informative only
once LoRA has moved.

Pairs: per-prompt pass × fail enumeration from harvest, capped at
max_pairs.

Threshold: 2σ ≈ 0.022 absolute final-pass-rate delta vs S1 baseline.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent / "phase14_s1"))
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
        return "", 0
    truncated_ids = []
    for tok_id in completion_ids:
        truncated_ids.append(tok_id)
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        if "\n" in decoded:
            break
    completion_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    if completion_text.endswith("\n"):
        completion_text = completion_text.rstrip("\n")
    return completion_text, len(truncated_ids)


def harvest_round(model, tokenizer, samples_per_prompt, seed_base,
                  max_new_tokens, temperature, top_p):
    records = []
    for ci, ch in enumerate(CHALLENGES):
        for j in range(samples_per_prompt):
            comp, n_tok = generate_and_score(
                model, tokenizer, ch["prompt"],
                max_new_tokens, temperature, top_p,
                seed_base + ci * 10000 + j,
            )
            if n_tok == 0:
                continue
            verdict = verify(ch["prompt"], comp, ch["suffix"])
            records.append({
                "challenge": ch["name"], "completion": comp,
                "n_tokens": n_tok, "verdict": verdict,
            })
    return records


def make_preference_pairs(records, max_per_prompt=4):
    """Per-prompt pass × fail enumeration. Returns list of
    (prompt, chosen, rejected). Cap pairs per prompt to keep batch
    sizes reasonable."""
    by_ch = defaultdict(lambda: {"pass": [], "fail": []})
    ch_prompt = {ch["name"]: ch["prompt"] for ch in CHALLENGES}
    for rec in records:
        bucket = "pass" if rec["verdict"] else "fail"
        by_ch[rec["challenge"]][bucket].append(rec["completion"])
    pairs = []
    for name, buckets in by_ch.items():
        passes, fails = buckets["pass"], buckets["fail"]
        if not passes or not fails:
            continue
        prompt = ch_prompt[name]
        n = min(len(passes), len(fails), max_per_prompt)
        for i in range(n):
            pairs.append((prompt, passes[i], fails[i]))
    return pairs


# ---------- SFT path ----------

class SftDataset(Dataset):
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


def sft_collate(batch, pad_id):
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


def sft_finetune(model, tokenizer, sft_pairs, steps, batch_size, lr, device):
    if not sft_pairs:
        return 0.0
    ds = SftDataset(sft_pairs, tokenizer)
    pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: sft_collate(b, pad_id))
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


# ---------- DPO / Hybrid path ----------

class DpoDataset(Dataset):
    def __init__(self, triples, tokenizer, max_len=128):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        prompt, chosen, rejected = self.triples[i]
        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        chosen_ids = self.tok(prompt + chosen + "\n", return_tensors="pt",
                              truncation=True, max_length=self.max_len).input_ids[0]
        rejected_ids = self.tok(prompt + rejected + "\n", return_tensors="pt",
                                truncation=True, max_length=self.max_len).input_ids[0]
        return {
            "prompt_len": prompt_ids.shape[0],
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
        }


def dpo_collate(batch, pad_id):
    max_c = max(b["chosen_ids"].shape[0] for b in batch)
    max_r = max(b["rejected_ids"].shape[0] for b in batch)
    n = len(batch)
    chosen_ids = torch.full((n, max_c), pad_id, dtype=torch.long)
    chosen_attn = torch.zeros((n, max_c), dtype=torch.long)
    chosen_labels = torch.full((n, max_c), -100, dtype=torch.long)
    rejected_ids = torch.full((n, max_r), pad_id, dtype=torch.long)
    rejected_attn = torch.zeros((n, max_r), dtype=torch.long)
    rejected_labels = torch.full((n, max_r), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        plen = b["prompt_len"]
        cn = b["chosen_ids"].shape[0]
        rn = b["rejected_ids"].shape[0]
        chosen_ids[i, :cn] = b["chosen_ids"]
        chosen_attn[i, :cn] = 1
        chosen_labels[i, plen:cn] = b["chosen_ids"][plen:cn]
        rejected_ids[i, :rn] = b["rejected_ids"]
        rejected_attn[i, :rn] = 1
        rejected_labels[i, plen:rn] = b["rejected_ids"][plen:rn]
    return {
        "chosen_ids": chosen_ids, "chosen_attn": chosen_attn, "chosen_labels": chosen_labels,
        "rejected_ids": rejected_ids, "rejected_attn": rejected_attn, "rejected_labels": rejected_labels,
    }


def sequence_logp(model, ids, attn, labels):
    """Sum log-prob of label tokens (positions where labels != -100)."""
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    gathered = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask.float()
    return gathered.sum(dim=1)  # [B]


def dpo_finetune(model, tokenizer, triples, steps, batch_size, lr, device,
                 beta, alpha):
    """Hybrid DPO: loss = (1-α)·DPO + α·SFT_chosen.
    α=1.0 → pure SFT on chosen; α=0.0 → pure DPO.
    Reference = PEFT base (LoRA disabled)."""
    if not triples:
        return 0.0
    ds = DpoDataset(triples, tokenizer)
    pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: dpo_collate(b, pad_id))
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
            # Policy logp (LoRA enabled)
            pol_chosen = sequence_logp(model, batch["chosen_ids"],
                                       batch["chosen_attn"], batch["chosen_labels"])
            pol_rejected = sequence_logp(model, batch["rejected_ids"],
                                         batch["rejected_attn"], batch["rejected_labels"])
            # Reference logp (LoRA disabled, no_grad)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen = sequence_logp(model, batch["chosen_ids"],
                                               batch["chosen_attn"], batch["chosen_labels"])
                    ref_rejected = sequence_logp(model, batch["rejected_ids"],
                                                 batch["rejected_attn"], batch["rejected_labels"])
            chosen_logratio = pol_chosen - ref_chosen
            rejected_logratio = pol_rejected - ref_rejected
            dpo_loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()
            # SFT chosen NLL: -mean over labelled tokens
            sft_loss = -(pol_chosen / batch["chosen_labels"].ne(-100).sum(dim=1).clamp(min=1).float()).mean()
            loss = (1.0 - alpha) * dpo_loss + alpha * sft_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            last = loss.item()
            step += 1
    model.eval()
    return last


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
    ap.add_argument("--dpo-mode", choices=["hybrid", "round0"], default="hybrid",
                    help="hybrid=(1-α)DPO+α·SFT all rounds; round0=pure DPO at r=0, SFT after")
    ap.add_argument("--alpha", type=float, default=0.3, help="SFT mixing weight (Phase 11 S5 best)")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--max-pairs-per-prompt", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase14_c3/run_{args.dpo_mode}_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P14C3] seed={args.seed} mode={args.dpo_mode} α={args.alpha} β={args.beta} device={device}")
    print(f"[P14C3] {len(CHALLENGES)} challenges")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P14C3] base loaded in {time.time() - t0:.1f}s")

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
        print(f"\n========== seed={args.seed} {args.dpo_mode} {label} ==========")
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

        # Decide which trainer to call this round
        ch_prompt = {ch["name"]: ch["prompt"] for ch in CHALLENGES}
        sft_pairs = [(ch_prompt[r_["challenge"]], r_["completion"])
                     for r_ in records if r_["verdict"]]
        pref_triples = make_preference_pairs(records, max_per_prompt=args.max_pairs_per_prompt)
        n_pairs = len(pref_triples)
        n_sft = len(sft_pairs)

        # Determine effective alpha for this round
        if args.dpo_mode == "hybrid":
            eff_alpha = args.alpha
            use_dpo = True
        elif args.dpo_mode == "round0":
            eff_alpha = 0.0 if r == 0 else 1.0  # pure DPO at r=0, pure SFT after
            use_dpo = (r == 0)
        else:
            raise ValueError(args.dpo_mode)

        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": rate,
            "n_sft_pairs": n_sft, "n_pref_pairs": n_pairs,
            "eff_alpha": eff_alpha, "used_dpo": use_dpo,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })

        if is_post:
            break

        # Train
        t = time.time()
        if use_dpo and n_pairs >= 2:
            loss = dpo_finetune(model, tokenizer, pref_triples,
                                steps=args.train_steps, batch_size=args.batch_size,
                                lr=args.lr, device=device,
                                beta=args.beta, alpha=eff_alpha)
            print(f"  DPO[α={eff_alpha:.2f} β={args.beta}]: {n_pairs} triples × "
                  f"{args.train_steps} steps, last_loss={loss:.3f}, "
                  f"{time.time() - t:.1f}s")
        elif n_sft >= 2:
            loss = sft_finetune(model, tokenizer, sft_pairs,
                                steps=args.train_steps, batch_size=args.batch_size,
                                lr=args.lr, device=device)
            print(f"  SFT: {n_sft} pairs × {args.train_steps} steps, "
                  f"last_loss={loss:.3f}, {time.time() - t:.1f}s")
        else:
            print(f"  [WARN] insufficient training data; skipping (n_pairs={n_pairs}, n_sft={n_sft})")

    out = {
        "model": args.model, "seed": args.seed, "dpo_mode": args.dpo_mode,
        "alpha": args.alpha, "beta": args.beta,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P14C3 seed={args.seed} {args.dpo_mode}] wrote {out_path}")
    print(f"\n=== summary: pass rate per round (seed={args.seed} {args.dpo_mode}) ===")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} "
              f"({h['n_pass']}/{h['n']})  pairs={h['n_pref_pairs']:3d}  α_eff={h['eff_alpha']:.2f}")


if __name__ == "__main__":
    main()
