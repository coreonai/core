"""Phase 16 S4 — Hybrid OPD+SFT vs SFT-only (P15 S1) and pure-OPD (P15 S2 + P16 S2)."""

import json
import statistics
from pathlib import Path

S4_DIR = Path(__file__).parent
S15_S1 = S4_DIR.parent / "phase15_s1"
S15_S2 = S4_DIR.parent / "phase15_s2"
S16_S2 = S4_DIR.parent / "phase16_s2"


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
    print(f"  {label:38s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
    return mu, sd


def main():
    hybrid = load(S4_DIR, "run_hybrid_*_seed*.json")
    sft = load(S15_S1, "run_seed*.json")
    fwd_opd = load(S15_S2, "run_opd_seed*.json")
    rev_opd = load(S16_S2, "run_revkl_seed*.json")
    if not hybrid:
        print("no hybrid run files yet")
        return
    print(f"\n=== Phase 16 S4 — Hybrid OPD+SFT vs all OPD/SFT predecessors ===\n")
    sft_mu, sft_sd = show(sft, "SFT only (Phase 15 S1)")
    fwd_mu, _ = show(fwd_opd, "Pure forward-KL OPD (Phase 15 S2)")
    rev_mu, _ = show(rev_opd, "Pure reverse-KL OPD (Phase 16 S2)")
    hyb_mu, hyb_sd = show(hybrid, f"Hybrid OPD+SFT α=0.3 reverse-KL (this commit)")

    print(f"\n=== Verdict ===")
    delta_sft = hyb_mu - sft_mu
    threshold = 2 * max(sft_sd, hyb_sd)
    print(f"  hybrid vs SFT:        Δ={delta_sft:+.3f}  2σ_max={threshold:.3f}")
    if abs(delta_sft) > threshold:
        v = "ROBUST WIN" if delta_sft > 0 else "ROBUST LOSS"
    else:
        v = "WITHIN NOISE"
    print(f"    → {v}")
    delta_rev = hyb_mu - rev_mu
    print(f"\n  hybrid vs pure rev-KL OPD: Δ={delta_rev:+.3f}")
    if delta_rev > 0.030:
        print(f"    → SFT anchor RESCUES OPD (substantially better than pure OPD)")
    elif delta_rev < -0.030:
        print(f"    → Hybrid worse than pure — surprising")
    else:
        print(f"    → Hybrid ≈ pure rev-KL OPD")


if __name__ == "__main__":
    main()
