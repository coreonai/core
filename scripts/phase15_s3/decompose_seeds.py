"""Phase 15 S3 — variance-axis-decomposed self-improve harness.

Phase 14 S1's σ=0.011 measurement used a single `--seed` argument to
control BOTH torch global RNG (LoRA delta_A init) AND the harvest
seed_base offset. Real noise is the sum of these two axes plus
temperature/checkpoint axes; the entangled measurement is a lower
bound.

This fork exposes the two axes as separate flags so we can decompose:
  - Fix --harvest-seed, vary --init-seed   →  σ_init  (LoRA init RNG only)
  - Fix --init-seed,    vary --harvest-seed → σ_harvest (sampling RNG only)
  - Vary both equally                       → σ_combined (= Phase 14 S1)

Same 25-problem set as Phase 14 S1 — cheap (~9 min/run).

Variance decomposition expectation:
  σ_combined² ≈ σ_init² + σ_harvest²
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "phase14_s1"))
from self_improve import harvest_round, lora_finetune  # noqa: E402
from problems import CHALLENGES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--init-seed", type=int, required=True,
                    help="torch.manual_seed for LoRA init (controls delta_A RNG)")
    ap.add_argument("--harvest-seed", type=int, required=True,
                    help="seed_base for harvest sampling (per-prompt RNG)")
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Init seed — controls torch global RNG at LoRA construction time
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase15_s3/"
        f"run_init{args.init_seed}_harvest{args.harvest_seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P15S3] init={args.init_seed} harvest={args.harvest_seed} device={device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P15S3] base loaded in {time.time() - t0:.1f}s")

    # LoRA init happens here — entirely controlled by torch.manual_seed above
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    history = []
    # IMPORTANT: harvest seed_base is independent of init_seed
    seed_base = args.harvest_seed * 1_000_000
    for r in range(args.rounds + 1):
        is_post = r == args.rounds
        label = f"round-{r}" if not is_post else f"final-{r}"
        print(f"\n========== init={args.init_seed} harvest={args.harvest_seed} {label} ==========")
        # phase14_s1 harvest_round uses (model, tokenizer, samples, seed_base, ...)
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
                             lr=args.lr, device=device)
        print(f"  LoRA-FT: {len(pairs)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "init_seed": args.init_seed,
        "harvest_seed": args.harvest_seed, "rounds": args.rounds,
        "samples": args.samples, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "train_steps": args.train_steps,
        "lr": args.lr, "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P15S3 init={args.init_seed} harvest={args.harvest_seed}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
