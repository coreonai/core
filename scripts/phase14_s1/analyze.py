"""Phase 14 S1 analyzer: aggregate 5-seed run JSONs, compute mean ±
std on (a) per-round pass rate, (b) final pass rate, (c) per-
challenge pass rates. Compare against K9 1M noise floor."""

import json
import statistics
from pathlib import Path

S1_DIR = Path(__file__).parent


def load_runs():
    runs = []
    for p in sorted(S1_DIR.glob("run_seed*.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def main():
    runs = load_runs()
    if not runs:
        print(f"no run_seed*.json files in {S1_DIR}")
        return

    print(f"\n=== Phase 14 S1 — Qwen substrate variance bound ({len(runs)} seeds) ===\n")
    n_rounds = max(len(r["history"]) for r in runs)
    n_challenges = max(len(r["history"][0]["per_challenge"]) for r in runs)
    print(f"Setup: {len(runs)} seeds, {n_rounds} round-snapshots per seed, {n_challenges} challenges\n")

    print("Per-round mean ± std pass rate:")
    for r_idx in range(n_rounds):
        per_seed = [
            run["history"][r_idx]["pass_rate"]
            for run in runs
            if r_idx < len(run["history"])
        ]
        if not per_seed:
            continue
        mu = statistics.mean(per_seed)
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        per_seed_s = ", ".join(f"{x:.3f}" for x in per_seed)
        label = runs[0]["history"][r_idx]["label"]
        print(f"  {label:10s}  {mu:.3f} ± {sd:.3f}  (per-seed: [{per_seed_s}])")

    final_rates = [run["history"][-1]["pass_rate"] for run in runs]
    final_mu = statistics.mean(final_rates)
    final_sd = statistics.stdev(final_rates) if len(final_rates) > 1 else 0.0
    print(f"\n  FINAL pass rate mean ± σ = {final_mu:.3f} ± {final_sd:.3f}")

    # Per-challenge: which problems are warm vs cold start?
    print("\nPer-challenge final pass rate (mean ± σ across seeds):")
    challenge_names = list(runs[0]["history"][-1]["per_challenge"].keys())
    saturated, headroom, cold_start = 0, 0, 0
    for name in challenge_names:
        per_seed_pass = []
        per_seed_total = []
        for run in runs:
            ch = run["history"][-1]["per_challenge"].get(name)
            if ch is None or ch["total"] == 0:
                continue
            per_seed_pass.append(ch["pass"] / ch["total"])
            per_seed_total.append(ch["total"])
        if not per_seed_pass:
            continue
        mu = statistics.mean(per_seed_pass)
        sd = statistics.stdev(per_seed_pass) if len(per_seed_pass) > 1 else 0.0
        bucket = "saturated" if mu >= 0.95 else "cold-start" if mu < 0.05 else "headroom"
        if bucket == "saturated": saturated += 1
        elif bucket == "cold-start": cold_start += 1
        else: headroom += 1
        print(f"  {name:30s}  {mu:.3f} ± {sd:.3f}  ({bucket})")

    print(f"\n=== Substrate noise floor verdict ===\n")
    print(f"  Final pass rate σ = {final_sd:.3f}")
    print(f"  K9 1M σ on final_eval/24 ≈ 0.142 (3.4/24, within-batch)")
    print(f"  K9 1M σ cross-batch ≈ 0.292 (7/24)")
    if final_sd < 0.05:
        print(f"  → ROBUST substrate. σ ≪ K9 noise floor, ready for Phase 14 C2/C3.")
    elif final_sd < 0.10:
        print(f"  → MARGINAL. σ comparable to K9. Larger benchmark may help.")
    else:
        print(f"  → NOISY. σ ≥ K9. Need richer benchmark or longer training.")
    print(f"\n  Problem-set composition: {saturated} saturated / {headroom} headroom / {cold_start} cold-start")
    print(f"    headroom problems are the ones that algorithmic comparisons can move")


if __name__ == "__main__":
    main()
