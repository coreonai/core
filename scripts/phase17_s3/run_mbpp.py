"""Phase 17 S3 — SFT self-improve on MBPP-100 substrate.

Reuses Phase 15 S1 self_improve harvest_round + lora_finetune but
swaps in MBPP CHALLENGES instead of HumanEval.
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

# Import MBPP problems FIRST so we can re-bind in sys.modules before
# self_improve transitively imports HumanEval problems.
sys.path.insert(0, str(Path(__file__).parent))
import problems as _mbpp_problems  # noqa: E402
CHALLENGES = _mbpp_problems.CHALLENGES

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round, lora_finetune  # noqa: E402


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
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase17_s3/run_mbpp_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P17S3-MBPP] seed={args.seed} n_challenges={len(CHALLENGES)}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P17S3-MBPP] base loaded in {time.time() - t0:.1f}s")

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
        print(f"\n========== seed={args.seed} MBPP {label} ==========")
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
        loss = lora_finetune(model, tokenizer, pairs,
                             steps=args.train_steps, batch_size=args.batch_size,
                             lr=args.lr, device=device)
        print(f"  LoRA-FT: {len(pairs)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed, "n_challenges": len(CHALLENGES),
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P17S3-MBPP seed={args.seed}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
