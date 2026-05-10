"""Phase 16 S2 — Reverse-KL OPD vs SFT-on-union (Phase 15 S1) and
forward-KL OPD (Phase 15 S2). Tests if Phase 15 S2's LOSS was
forward-KL-specific.
"""

import json
import statistics
from pathlib import Path

S2_DIR = Path(__file__).parent
S15_S1_DIR = S2_DIR.parent / "phase15_s1"
S15_S2_DIR = S2_DIR.parent / "phase15_s2"


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
    print(f"  {label:30s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
    return mu, sd, finals


def main():
    rev = load(S2_DIR, "run_revkl_seed*.json")
    fwd = load(S15_S2_DIR, "run_opd_seed*.json")
    sft = load(S15_S1_DIR, "run_seed*.json")
    if not (rev and fwd and sft):
        print(f"missing runs: rev={len(rev)} fwd={len(fwd)} sft={len(sft)}")
        return

    print(f"\n=== Phase 16 S2 — Reverse-KL OPD vs forward-KL vs SFT ===\n")
    sft_mu, sft_sd, _ = show(sft, "SFT (Phase 15 S1)")
    fwd_mu, fwd_sd, _ = show(fwd, "Forward-KL OPD (Phase 15 S2)")
    rev_mu, rev_sd, _ = show(rev, "Reverse-KL OPD (this commit)")

    print(f"\n=== Verdict ===")
    delta_rev_sft = rev_mu - sft_mu
    threshold = 2 * max(sft_sd, rev_sd)
    print(f"  rev vs SFT:    Δ={delta_rev_sft:+.3f}  2σ_max={threshold:.3f}")
    if abs(delta_rev_sft) > threshold:
        v = "ROBUST WIN" if delta_rev_sft > 0 else "ROBUST LOSS"
    else:
        v = "WITHIN NOISE"
    print(f"    → reverse-KL {v} vs SFT")

    delta_rev_fwd = rev_mu - fwd_mu
    print(f"\n  rev vs fwd:    Δ={delta_rev_fwd:+.3f}")
    if delta_rev_fwd > 0.030:
        print(f"    → reverse-KL substantially better than forward-KL "
              f"(Phase 15 S2 LOSS was KL-direction-specific)")
    elif delta_rev_fwd < -0.030:
        print(f"    → reverse-KL worse than forward-KL (unexpected)")
    else:
        print(f"    → reverse-KL ≈ forward-KL (KL direction doesn't matter)")


if __name__ == "__main__":
    main()
