"""Phase 17 S2 — SFT with label smoothing as overfitting regularizer.

Phase 15 S1 found lift bimodality (LIFTED 0/3 vs FLAT 1/4) caused by
overfitting: FLAT seeds achieve LOWER LoRA-FT training loss but
WORSE generalization. Label smoothing is a textbook regularizer
against overfitting on hard targets.

This is a mechanism-targeted primitive (not a paper port). Cheap to
test, well-understood, and directly addresses the documented Phase
15 S1 mechanism.

Implementation: instead of HF's default cross-entropy on hard labels,
use `torch.nn.functional.cross_entropy(..., label_smoothing=α)`. With
α=0.1, the target distribution is a mix of one-hot (90%) + uniform
(10% / vocab_size).

Decision gate (Phase 16 S1 σ=0.031, 2σ=0.062):
- Δ_LS-SFT > +0.062 → robust positive lift (label smoothing rescues
  overfitting mechanism)
- σ_LS < σ_SFT/2 (= 0.016) → variance-reduction win (LIFTED/FLAT
  bimodality eliminated)
- Both fail → label smoothing doesn't address mechanism at this
  scale
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round, SftDataset, collate  # noqa: E402
from problems import CHALLENGES  # noqa: E402


def label_smoothed_loss(logits, labels, label_smoothing):
    """Standard label-smoothed CE on a [B, T, V] logits + [B, T]
    labels tensor with -100 ignored (causal-LM convention)."""
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Flatten
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def lora_finetune_label_smooth(model, tokenizer, pairs, steps, batch_size,
                                lr, device, label_smoothing):
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
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss = label_smoothed_loss(out.logits, batch["labels"],
                                        label_smoothing)
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
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase17_s2/"
        f"run_ls{args.label_smoothing}_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P17S2] seed={args.seed} label_smoothing={args.label_smoothing}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P17S2] base loaded in {time.time() - t0:.1f}s")

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
        print(f"\n========== seed={args.seed} ls={args.label_smoothing} {label} ==========")
        records = harvest_round(
            model, tokenizer, CHALLENGES, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
            args.verify_timeout,
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
        print(f"  total={n} pass={n_pass} ({rate:.3f})  {time.time() - t_round:.1f}s")
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
        loss = lora_finetune_label_smooth(
            model, tokenizer, pairs,
            steps=args.train_steps, batch_size=args.batch_size,
            lr=args.lr, device=device,
            label_smoothing=args.label_smoothing,
        )
        print(f"  LoRA-FT[ls={args.label_smoothing}]: {len(pairs)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed,
        "label_smoothing": args.label_smoothing,
        "rounds": args.rounds, "samples": args.samples,
        "n_challenges": len(CHALLENGES),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P17S2 seed={args.seed} ls={args.label_smoothing}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
