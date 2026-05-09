"""Phase 15 S1 analyzer: aggregate 5-seed run JSONs at HumanEval
substrate, compute per-round mean ± σ, per-challenge bucketing
(saturated / headroom / cold-start), and decision verdict for
substrate qualification.

Decision gate: σ_final ≤ 0.03 AND ≤ 50% saturated → ROBUST.
"""

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

    n_rounds = max(len(r["history"]) for r in runs)
    n_challenges = max(len(r["history"][0]["per_challenge"]) for r in runs)
    print(f"\n=== Phase 15 S1 — HumanEval substrate ({len(runs)} seeds, "
          f"{n_challenges} problems) ===\n")

    print("Per-round mean ± σ pass rate:")
    final_rates = []
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
        print(f"  {label:10s}  {mu:.3f} ± {sd:.3f}  [{per_seed_s}]")
        if r_idx == n_rounds - 1:
            final_rates = per_seed

    final_mu = statistics.mean(final_rates)
    final_sd = statistics.stdev(final_rates) if len(final_rates) > 1 else 0.0
    print(f"\n  FINAL pass rate mean ± σ = {final_mu:.3f} ± {final_sd:.3f}")

    # Per-challenge bucketing
    print("\nPer-challenge final pass rate distribution:")
    challenge_names = list(runs[0]["history"][-1]["per_challenge"].keys())
    saturated, headroom, cold_start = 0, 0, 0
    for name in challenge_names:
        per_seed_pass = []
        for run in runs:
            ch = run["history"][-1]["per_challenge"].get(name)
            if ch is None or ch["total"] == 0:
                continue
            per_seed_pass.append(ch["pass"] / ch["total"])
        if not per_seed_pass:
            continue
        mu = statistics.mean(per_seed_pass)
        if mu >= 0.95:
            saturated += 1
        elif mu < 0.05:
            cold_start += 1
        else:
            headroom += 1

    total = saturated + headroom + cold_start
    print(f"  Saturated  (mu >= 0.95): {saturated:3d}  ({saturated/total*100:.0f}%)")
    print(f"  Headroom   (0.05 <= mu < 0.95): {headroom:3d}  ({headroom/total*100:.0f}%)")
    print(f"  Cold-start (mu < 0.05): {cold_start:3d}  ({cold_start/total*100:.0f}%)")

    # Substrate verdict
    print(f"\n=== Substrate qualification ===")
    print(f"  Final σ = {final_sd:.3f}  (target ≤ 0.03)")
    print(f"  Saturated fraction = {saturated/total:.2f}  (target ≤ 0.50)")
    sigma_ok = final_sd <= 0.03
    sat_ok = saturated / total <= 0.50
    if sigma_ok and sat_ok:
        verdict = "ROBUST — qualified for Phase 15 algorithmic comparisons"
    elif sigma_ok:
        verdict = f"NOISY-OK but TOO SATURATED ({saturated/total*100:.0f}%) — needs harder problems"
    elif sat_ok:
        verdict = f"HEADROOM-OK but NOISY (σ={final_sd:.3f}) — more samples or train-steps"
    else:
        verdict = "FAIL — both σ too high and substrate too saturated"
    print(f"  → {verdict}")

    # Phase 14 comparison
    print(f"\n=== Comparison vs Phase 14 substrate ===")
    print(f"  Phase 14 (25 problems): final σ=0.011, saturation 84%")
    print(f"  Phase 15 (164 HumanEval): final σ={final_sd:.3f}, saturation {saturated/total*100:.0f}%")


if __name__ == "__main__":
    main()
