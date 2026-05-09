"""Phase 15 S4 analyzer: Muon at HumanEval substrate vs Phase 15 S1
SFT (AdamW) baseline. Falsifier test for Phase 14 C2's Muon-LOSES
verdict — does it generalize from saturated Phase 14 to headroom-rich
Phase 15?

Decision gate (using S1's σ=0.041): 2σ = 0.082 absolute delta.
"""

import json
import statistics
from pathlib import Path

S4_DIR = Path(__file__).parent
S1_DIR = S4_DIR.parent / "phase15_s1"


def load_runs(d, glob):
    return [json.loads(p.read_text()) for p in sorted(d.glob(glob))]


def round_stats(runs, r_idx):
    rates = [r["history"][r_idx]["pass_rate"] for r in runs
             if r_idx < len(r["history"])]
    if not rates:
        return None
    mu = statistics.mean(rates)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    return mu, sd, rates


def aggregate(runs, label):
    n_rounds = max(len(r["history"]) for r in runs)
    print(f"\n--- {label} ({len(runs)} seeds) ---")
    final = None
    for r_idx in range(n_rounds):
        s = round_stats(runs, r_idx)
        if s is None:
            continue
        mu, sd, rates = s
        rs = ", ".join(f"{x:.3f}" for x in rates)
        rlabel = runs[0]["history"][r_idx]["label"]
        print(f"  {rlabel:10s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
        if r_idx == n_rounds - 1:
            final = (mu, sd, rates)
    return final


def main():
    sft_runs = load_runs(S1_DIR, "run_seed*.json")
    muon_runs = load_runs(S4_DIR, "run_muon_seed*.json")
    if not (sft_runs and muon_runs):
        print("missing one of: SFT (S1) / Muon (S4) run files")
        return

    print(f"\n=== Phase 15 S4 — Muon vs AdamW LoRA at HumanEval substrate ===")
    print(f"SFT (S1, AdamW): {len(sft_runs)} seeds   Muon: {len(muon_runs)} seeds\n")

    sft_final = aggregate(sft_runs, "AdamW (Phase 15 S1)")
    muon_final = aggregate(muon_runs, "Muon")
    if not (sft_final and muon_final):
        print("aggregate returned None")
        return

    sft_mu, sft_sd, _ = sft_final
    muon_mu, muon_sd, _ = muon_final
    delta = muon_mu - sft_mu
    sigma_max = max(sft_sd, muon_sd)
    threshold = 2 * sigma_max

    print(f"\n=== Final pass rate comparison ===")
    print(f"  AdamW: {sft_mu:.3f} ± {sft_sd:.3f}")
    print(f"  Muon : {muon_mu:.3f} ± {muon_sd:.3f}")
    print(f"  Δ    = {delta:+.3f}")
    print(f"  2σ_max = {threshold:.3f}")
    if abs(delta) > threshold:
        v = "ROBUST WIN for Muon" if delta > 0 else "ROBUST LOSS for Muon"
    else:
        v = "WITHIN NOISE"
    print(f"  → {v}")

    # Phase 14 C2 comparison
    print(f"\n=== Phase 14 C2 (saturated 25 problems) reminder ===")
    print(f"  AdamW 0.851 ± 0.011   Muon 0.759 ± 0.004   Δ = -0.092 (4× threshold)")
    print(f"  Phase 14 verdict: ROBUST LOSS for Muon")
    print(f"\nPhase 15 S4 verdict ({delta:+.3f}) {'CONFIRMS' if delta < -threshold else 'DOES NOT CONFIRM' if delta > -threshold else 'ambiguous on'} Phase 14 C2's generalization.")


if __name__ == "__main__":
    main()
