"""Phase 17 S9 — pass@k at base Qwen on MBPP-100 substrate.

S6 found pass@10 = 0.524 on HumanEval (vs pass@1 = 0.216). Does the
inference-time scaling effect generalize to MBPP? S3 already showed
MBPP is more learnable for SFT (mean lift +0.146 vs HumanEval
+0.032). Higher base pass@1 would predict similar pass@k advantage.

Single seed, k=10. Cross-substrate validation of S6 finding.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Important: load MBPP problems FIRST so they win the sys.modules
# cache when phase15_s1.self_improve transitively imports `problems`.
sys.path.insert(0, str(Path(__file__).parent.parent / "phase17_s3"))
import problems as _mbpp_problems  # noqa: E402
CHALLENGES = _mbpp_problems.CHALLENGES

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import generate_completion, verify  # noqa: E402


def pass_at_k_unbiased(n_total, n_correct, k):
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_total - n_correct, k) / math.comb(n_total, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase17_s9/run_passk_mbpp_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P17S9-passk-mbpp] seed={args.seed} k={args.k} n_challenges={len(CHALLENGES)}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"[P17S9-passk-mbpp] base loaded in {time.time() - t0:.1f}s")

    per_challenge = {}
    seed_base = args.seed * 1_000_000
    t_start = time.time()
    for ci, ch in enumerate(CHALLENGES):
        passes = 0
        for j in range(args.k):
            comp, n_tok = generate_completion(
                model, tokenizer, ch["prompt"],
                args.max_new_tokens, args.temperature, args.top_p,
                seed_base + ci * 10000 + j,
            )
            if n_tok == 0:
                continue
            if verify(ch["prompt"], comp, ch["suffix"], timeout=args.verify_timeout):
                passes += 1
        per_challenge[ch["name"]] = {"passes": passes, "k": args.k}
        if (ci + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{ci+1}/{len(CHALLENGES)}] {elapsed:.0f}s elapsed")

    pass_at_k_raw = sum(1 for v in per_challenge.values() if v["passes"] > 0) / len(per_challenge)
    total_passes = sum(v["passes"] for v in per_challenge.values())
    pass_at_1_raw = total_passes / (args.k * len(per_challenge))

    estimates = {}
    for k_eval in [1, 2, 5, 10]:
        if k_eval > args.k:
            continue
        per_problem = [pass_at_k_unbiased(args.k, v["passes"], k_eval)
                       for v in per_challenge.values()]
        estimates[f"pass@{k_eval}"] = sum(per_problem) / len(per_problem)

    print(f"\n=== MBPP pass@k results (base Qwen, no training) ===")
    print(f"  pass@1 raw:    {pass_at_1_raw:.3f}")
    print(f"  pass@k raw:    {pass_at_k_raw:.3f}")
    print(f"  Unbiased:")
    for kn, kv in estimates.items():
        print(f"    {kn:8s} = {kv:.3f}")

    out = {
        "model": args.model, "seed": args.seed, "k": args.k,
        "substrate": "mbpp_100", "n_challenges": len(per_challenge),
        "pass_at_1_raw": pass_at_1_raw,
        "pass_at_k_raw": pass_at_k_raw,
        "estimates": estimates,
        "per_challenge": per_challenge,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P17S9 seed={args.seed}] wrote {out_path}")
    print(f"\nCross-substrate comparison:")
    print(f"  HumanEval (S6): pass@1 = 0.216, pass@10 = 0.524, Δ = +0.308")
    print(f"  MBPP    (S9): pass@1 = {estimates.get('pass@1', 0):.3f}, "
          f"pass@10 = {estimates.get('pass@10', 0):.3f}, "
          f"Δ = {estimates.get('pass@10', 0) - estimates.get('pass@1', 0):+.3f}")


if __name__ == "__main__":
    main()
