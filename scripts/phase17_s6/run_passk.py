"""Phase 17 S6 — pass@k inference-only baseline at HumanEval-164.

Phases 14-16 measured single-sample pass-rate (essentially pass@1 at
T=0.8). Inference-time techniques are an orthogonal axis we haven't
explored. pass@k is the cheapest baseline: generate k samples per
prompt, count problem as solved if ≥1 sample passes the verifier.

This run uses BASE Qwen-Coder-0.5B (no LoRA, no training) — pure
inference. Result is the headroom achievable by inference scaling
alone, before any training intervention.

For comparison:
  Phase 15 S1 SFT mean (samples=3, scored as round-0 fraction): ~0.213
  This is essentially pass@1 at T=0.8.

  pass@1: count problems passing on average over k samples
          = mean(any pass in k samples) / k... actually no.
          pass@1 is "what fraction of single-sample attempts pass" —
          our Phase 15 round-0 0.213 IS pass@1.
  pass@k: "what fraction of problems have ≥1 pass in k samples"
          = strictly ≥ pass@1 by definition.

Decision gate:
- pass@10 - pass@1 > 0.062 (Phase 16 2σ): real headroom from
  inference scaling. Worth combining with training.
- pass@10 - pass@1 ≤ 0.062: inference scaling has limited headroom
  at this base model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import generate_completion, verify  # noqa: E402
from problems import CHALLENGES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--k", type=int, default=10, help="samples per prompt")
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
        f"/raid/users/paul/workLLM/scripts/phase17_s6/run_passk_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P17S6-passk] seed={args.seed} k={args.k} model={args.model}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"[P17S6-passk] model loaded in {time.time() - t0:.1f}s")

    # Per-challenge: track which of k attempts passed
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
            est_total = elapsed * len(CHALLENGES) / (ci + 1)
            print(f"  [{ci+1}/{len(CHALLENGES)}] {elapsed:.0f}s elapsed, "
                  f"~{est_total - elapsed:.0f}s remaining")

    # pass@k metric: fraction of problems with at least 1 pass
    pass_at_k = sum(1 for v in per_challenge.values() if v["passes"] > 0) / len(per_challenge)
    # pass@1 estimator: fraction of all (problem × sample) pairs that pass
    total_attempts = sum(v["k"] for v in per_challenge.values())
    total_passes = sum(v["passes"] for v in per_challenge.values())
    pass_at_1 = total_passes / total_attempts

    # Per-k variants: pass@1, pass@2, pass@5, pass@10 via unbiased estimator
    # pass@k_unbiased = mean over problems of [1 - C(k-c, k) / C(k, k)] when c < k passes
    import math
    def pass_at_k_unbiased(n_total, n_correct, k):
        if n_total - n_correct < k:
            return 1.0
        return 1.0 - math.comb(n_total - n_correct, k) / math.comb(n_total, k)

    estimates = {}
    for k_eval in [1, 2, 5, 10]:
        if k_eval > args.k:
            continue
        per_problem = [pass_at_k_unbiased(args.k, v["passes"], k_eval)
                        for v in per_challenge.values()]
        estimates[f"pass@{k_eval}"] = sum(per_problem) / len(per_problem)

    print(f"\n=== pass@k results (base model, no training) ===")
    print(f"  pass@1 (raw):      {pass_at_1:.3f}")
    print(f"  pass@k (k={args.k}, ≥1 pass): {pass_at_k:.3f}")
    print(f"  Unbiased estimators:")
    for name, val in estimates.items():
        print(f"    {name:8s} = {val:.3f}")

    out = {
        "model": args.model, "seed": args.seed, "k": args.k,
        "n_challenges": len(per_challenge),
        "pass_at_1_raw": pass_at_1,
        "pass_at_k_raw": pass_at_k,
        "estimates": estimates,
        "per_challenge": per_challenge,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P17S6-passk seed={args.seed}] wrote {out_path}")


if __name__ == "__main__":
    main()
