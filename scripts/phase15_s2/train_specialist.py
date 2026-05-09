"""Phase 15 S2 — train one SFT specialist on one subset of HumanEval.

Reuses Phase 15 S1 self-improve harness but constrains the challenge
list to a single routing subset. Saves the LoRA adapter so the OPD
student trainer can load it as a frozen teacher.

Usage:
  python train_specialist.py --subset strings --seed 99 \\
    --out-adapter checkpoints/phase15_s2/specialist_strings
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round, lora_finetune  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from routing import SUBSETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--subset", required=True, choices=list(SUBSETS.keys()))
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out-adapter", required=True,
                    help="Directory to save_pretrained the trained LoRA adapter")
    ap.add_argument("--out-meta", default=None,
                    help="JSON path for metadata (default: <out-adapter>/meta.json)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    challenges = SUBSETS[args.subset]
    out_adapter = Path(args.out_adapter)
    out_adapter.mkdir(parents=True, exist_ok=True)
    out_meta = Path(args.out_meta) if args.out_meta else (out_adapter / "meta.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P15S2-spec] subset={args.subset} ({len(challenges)} problems) "
          f"seed={args.seed} device={device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P15S2-spec] base loaded in {time.time() - t0:.1f}s")

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
        print(f"\n========== {args.subset} {label} ==========")
        records = harvest_round(
            model, tokenizer, challenges, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
            args.verify_timeout,
        )
        seed_base += 100 * len(challenges)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        rate = n_pass / n if n else 0.0
        print(f"  total={n} pass={n_pass} ({rate:.3f})  {time.time() - t_round:.1f}s")
        history.append({"label": label, "n": n, "n_pass": n_pass, "pass_rate": rate})
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

    # Save the trained adapter
    model.save_pretrained(str(out_adapter))
    print(f"\n[P15S2-spec] saved adapter to {out_adapter}")

    meta = {
        "subset": args.subset,
        "n_challenges": len(challenges),
        "challenge_names": [ch["name"] for ch in challenges],
        "model": args.model, "seed": args.seed,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[P15S2-spec] wrote {out_meta}")


if __name__ == "__main__":
    main()
