"""Phase 16 S3 — Muon vs AdamW at LoRA r=64 vs Phase 15 r=16 baselines.

Tests whether Phase 14 C2 + Phase 15 S4's Muon-LOSS verdict is
rank-specific (r=16 bottleneck) or rank-independent.
"""

import json
import statistics
from pathlib import Path

S3_DIR = Path(__file__).parent
S15_S1_DIR = S3_DIR.parent / "phase15_s1"  # AdamW r=16
S15_S4_DIR = S3_DIR.parent / "phase15_s4"  # Muon r=16


def load(d, glob):
    return [json.loads(p.read_text()) for p in sorted(d.glob(glob))]


def mu_sd(xs):
    if len(xs) < 2:
        return statistics.mean(xs), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def show(runs, label):
    finals = [r["history"][-1]["pass_rate"] for r in runs]
    mu, sd = mu_sd(finals)
    rs = ", ".join(f"{x:.3f}" for x in finals)
    print(f"  {label:32s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
    return mu, sd


def main():
    muon_r64 = load(S3_DIR, "run_muon_r64_seed*.json")
    adam_r64 = load(S3_DIR, "run_adam_r64_seed*.json")
    adam_r16 = load(S15_S1_DIR, "run_seed*.json")
    muon_r16 = load(S15_S4_DIR, "run_muon_seed*.json")
    if not (muon_r64 and adam_r64 and adam_r16 and muon_r16):
        print(f"missing runs: muon_r64={len(muon_r64)} adam_r64={len(adam_r64)} "
              f"adam_r16={len(adam_r16)} muon_r16={len(muon_r16)}")
        return

    print(f"\n=== Phase 16 S3 — Muon at higher LoRA rank ===\n")
    am16 = show(adam_r16, "AdamW r=16 (Phase 15 S1)")
    mm16 = show(muon_r16, "Muon  r=16 (Phase 15 S4)")
    am64 = show(adam_r64, "AdamW r=64 (this commit)")
    mm64 = show(muon_r64, "Muon  r=64 (this commit)")

    print(f"\n=== Δ within rank ===")
    print(f"  r=16: Muon - AdamW = {mm16[0] - am16[0]:+.3f}  (Phase 15 S4 verdict: LOSS)")
    print(f"  r=64: Muon - AdamW = {mm64[0] - am64[0]:+.3f}")

    print(f"\n=== Δ across rank (does higher rank help?) ===")
    print(f"  AdamW: r=64 - r=16 = {am64[0] - am16[0]:+.3f}")
    print(f"  Muon : r=64 - r=16 = {mm64[0] - mm16[0]:+.3f}")

    print(f"\n=== Verdict ===")
    delta_r64 = mm64[0] - am64[0]
    threshold = 2 * max(am64[1], mm64[1])
    print(f"  r=64 Muon vs AdamW Δ = {delta_r64:+.3f}, 2σ_max = {threshold:.3f}")
    if abs(delta_r64) > threshold:
        v = "ROBUST WIN for Muon" if delta_r64 > 0 else "ROBUST LOSS for Muon"
    else:
        v = "WITHIN NOISE"
    print(f"  → r=64: {v}")

    print(f"\n  Cross-rank generalization:")
    if mm64[0] - am64[0] < -0.030:
        print(f"    Muon LOSES at both r=16 and r=64. Mechanism is rank-independent.")
        print(f"    Phase 14 C2 + Phase 15 S4 verdict not rescued by higher rank.")
    elif mm64[0] - am64[0] > +0.030:
        print(f"    Muon WINS at r=64 (LOSES at r=16). Mechanism IS rank-specific —")
        print(f"    NS orthogonalization needs sufficient capacity to express its update")
        print(f"    direction. r=16 is the bottleneck; r=64 (4× capacity) unlocks.")
    else:
        print(f"    r=64 Muon comparable to AdamW. Phase 14 C2 verdict scoped to r ≤ 16.")


if __name__ == "__main__":
    main()
