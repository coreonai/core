"""Phase 17 S7b — SFT train at HumanEval + pass@k eval (post-training).

Tests whether SFT-trained model retains the multi-modal sample
diversity that gave base Qwen its 0.524 pass@10 (Phase 17 S6). If
training collapses diversity (Phase 15 S1's overfitting mechanism),
SFT pass@10 << base pass@10. If training adds capability, SFT
pass@10 > base pass@10.

Single seed (proof-of-concept). If finding is dramatic, expand to 5.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round, lora_finetune, generate_completion, verify  # noqa: E402
from problems import CHALLENGES  # noqa: E402


def pass_at_k_unbiased(n_total, n_correct, k):
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_total - n_correct, k) / math.comb(n_total, k)


def passk_eval(model, tokenizer, challenges, k, seed_base, max_new_tokens,
               temperature, top_p, verify_timeout):
    """Run k samples per challenge, return per-challenge pass counts."""
    per_challenge = {}
    for ci, ch in enumerate(challenges):
        passes = 0
        for j in range(k):
            comp, n_tok = generate_completion(
                model, tokenizer, ch["prompt"],
                max_new_tokens, temperature, top_p,
                seed_base + ci * 10000 + j,
            )
            if n_tok == 0:
                continue
            if verify(ch["prompt"], comp, ch["suffix"], timeout=verify_timeout):
                passes += 1
        per_challenge[ch["name"]] = {"passes": passes, "k": k}
    return per_challenge


def passk_estimates(per_challenge, k_eval_list):
    out = {}
    for k_eval in k_eval_list:
        if k_eval > next(iter(per_challenge.values()))["k"]:
            continue
        per_problem = [pass_at_k_unbiased(v["k"], v["passes"], k_eval)
                       for v in per_challenge.values()]
        out[f"pass@{k_eval}"] = sum(per_problem) / len(per_problem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--samples", type=int, default=6, help="harvest samples")
    ap.add_argument("--passk-k", type=int, default=10, help="post-training pass@k k")
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
        f"/raid/users/paul/workLLM/scripts/phase17_s7/"
        f"run_passk_sft_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P17S7b] seed={args.seed} samples={args.samples} passk_k={args.passk_k}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    print(f"[P17S7b] base loaded in {time.time() - t0:.1f}s")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # Step 1: harvest round 0
    seed_base = args.seed * 1_000_000
    print(f"\n========== seed={args.seed} round-0 harvest ==========")
    t = time.time()
    records = harvest_round(
        model, tokenizer, CHALLENGES, args.samples, seed_base,
        args.max_new_tokens, args.temperature, args.top_p, args.verify_timeout,
    )
    seed_base += 100 * len(CHALLENGES)
    n = len(records)
    n_pass = sum(1 for r in records if r["verdict"])
    r0_rate = n_pass / n
    print(f"  total={n} pass={n_pass} ({r0_rate:.3f})  {time.time() - t:.1f}s")

    # Step 2: train SFT
    ch_prompt = {ch["name"]: ch["prompt"] for ch in CHALLENGES}
    pairs = [(ch_prompt[r["challenge"]], r["completion"])
             for r in records if r["verdict"]]
    print(f"\n========== seed={args.seed} LoRA-FT ({len(pairs)} pairs) ==========")
    t = time.time()
    loss = lora_finetune(model, tokenizer, pairs,
                         steps=args.train_steps, batch_size=args.batch_size,
                         lr=args.lr, device=device)
    print(f"  last_loss={loss:.3f} {time.time() - t:.1f}s")

    # Step 3: pass@k eval on SFT-trained model
    print(f"\n========== seed={args.seed} pass@{args.passk_k} eval (SFT-trained) ==========")
    t = time.time()
    sft_per_ch = passk_eval(model, tokenizer, CHALLENGES, args.passk_k, seed_base,
                             args.max_new_tokens, args.temperature, args.top_p,
                             args.verify_timeout)
    sft_passk = passk_estimates(sft_per_ch, [1, 2, 5, 10])
    print(f"  SFT model pass@k results:")
    for kn, kv in sft_passk.items():
        print(f"    {kn:8s} = {kv:.3f}")
    print(f"  ({time.time() - t:.1f}s)")

    out = {
        "model": args.model, "seed": args.seed,
        "samples": args.samples, "passk_k": args.passk_k,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "round_0_pass_rate": r0_rate,
        "n_chosen_pairs": len(pairs),
        "sft_passk": sft_passk,
        "sft_per_challenge": sft_per_ch,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P17S7b seed={args.seed}] wrote {out_path}")
    print(f"\nComparison vs base Qwen pass@k (Phase 17 S6):")
    print(f"  base pass@1  = 0.216  vs SFT pass@1  = {sft_passk.get('pass@1', 0):.3f}")
    print(f"  base pass@10 = 0.524  vs SFT pass@10 = {sft_passk.get('pass@10', 0):.3f}")


if __name__ == "__main__":
    main()
