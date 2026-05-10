"""Phase 16 S1 — samples=6 substrate σ measurement.

Validates Phase 15 S3b's CLT prediction: doubling samples-per-prompt
should roughly halve σ_harvest. Phase 15 S1 used samples=3, σ=0.041.
Predicted samples=6 → σ ≈ 0.029.
"""

import json
import statistics
from pathlib import Path

S1_DIR = Path(__file__).parent
S15_S1_DIR = S1_DIR.parent / "phase15_s1"


def load(d, glob):
    return [json.loads(p.read_text()) for p in sorted(d.glob(glob))]


def mu_sd(xs):
    if len(xs) < 2:
        return statistics.mean(xs), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate(runs, label):
    n = max(len(r["history"]) for r in runs)
    print(f"\n--- {label} ({len(runs)} seeds) ---")
    for r_idx in range(n):
        rates = [r["history"][r_idx]["pass_rate"] for r in runs]
        mu, sd = mu_sd(rates)
        rs = ", ".join(f"{x:.3f}" for x in rates)
        print(f"  {runs[0]['history'][r_idx]['label']:10s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
    return mu_sd([r["history"][-1]["pass_rate"] for r in runs])


def main():
    s6 = load(S1_DIR, "run_s6_seed*.json")
    s3 = load(S15_S1_DIR, "run_seed*.json")
    if not (s6 and s3):
        print(f"missing runs (s6={len(s6)}, s3={len(s3)})")
        return
    print(f"\n=== Phase 16 S1 — samples=6 substrate σ ===\n")
    mu6, sd6 = aggregate(s6, "Phase 16 S1 — samples=6")
    mu3, sd3 = aggregate(s3, "Phase 15 S1 — samples=3 (baseline)")
    print(f"\n=== σ comparison ===")
    print(f"  samples=3: σ={sd3:.4f}")
    print(f"  samples=6: σ={sd6:.4f}")
    if sd3 > 0:
        ratio = sd6 / sd3
        clt_pred = 1.0 / (2 ** 0.5)  # ~0.71
        print(f"  Ratio σ_6/σ_3 = {ratio:.2f}  (CLT prediction: {clt_pred:.2f}, ~0.71)")
        if ratio < 0.85:
            print(f"  → S3b prediction VALIDATED (σ reduces with more samples)")
            if ratio < 0.6:
                print(f"  → Reduction stronger than CLT — per-prompt samples may be"
                      f" anti-correlated, helping more")
        elif ratio > 1.15:
            print(f"  → σ INCREASED — unexpected, mechanism review needed")
        else:
            print(f"  → σ reduction milder than CLT prediction")
    print(f"\n  Δ_mean = {mu6 - mu3:+.3f} (samples=6 mean shift vs samples=3)")
    print(f"\n=== Threshold update for Phase 16+ ===")
    print(f"  New 2σ threshold = {2*sd6:.3f}  (was {2*sd3:.3f} at Phase 15)")


if __name__ == "__main__":
    main()
