"""Phase 14 C3 analyzer: hybrid SFT+DPO and round-0-only DPO vs SFT
baseline (Phase 14 S1) at Qwen substrate.

Loads:
  scripts/phase14_c3/run_hybrid_seed*.json
  scripts/phase14_c3/run_round0_seed*.json
  scripts/phase14_s1/run_seed*.json   (SFT baseline)

Reports per-arm round mean ± σ, final pass-rate, Δ vs SFT, and 2σ
significance verdict.
"""

import json
import statistics
from pathlib import Path

C3_DIR = Path(__file__).parent
S1_DIR = C3_DIR.parent / "phase14_s1"


def load_runs(d, glob):
    return [json.loads(p.read_text()) for p in sorted(d.glob(glob))]


def round_stats(runs, r_idx):
    rates = [run["history"][r_idx]["pass_rate"]
             for run in runs if r_idx < len(run["history"])]
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
        # Show pair counts when available
        pair_info = ""
        ph = runs[0]["history"][r_idx]
        if "n_pref_pairs" in ph:
            pair_info = f"  pref={ph['n_pref_pairs']}/sft={ph['n_sft_pairs']}"
        print(f"  {rlabel:10s}  {mu:.3f} ± {sd:.3f}  [{rs}]{pair_info}")
        if r_idx == n_rounds - 1:
            final = (mu, sd, rates)
    return final


def main():
    sft_runs = load_runs(S1_DIR, "run_seed*.json")
    hybrid_runs = load_runs(C3_DIR, "run_hybrid_seed*.json")
    round0_runs = load_runs(C3_DIR, "run_round0_seed*.json")
    if not (sft_runs and hybrid_runs and round0_runs):
        print("missing one of: SFT/hybrid/round0 run files")
        return

    print(f"\n=== Phase 14 C3 — DPO variants vs SFT at Qwen substrate ===")
    print(f"SFT (S1): {len(sft_runs)}  Hybrid: {len(hybrid_runs)}  Round0: {len(round0_runs)}\n")

    sft_final = aggregate(sft_runs, "SFT (Phase 14 S1 baseline)")
    hyb_final = aggregate(hybrid_runs, f"Hybrid SFT+DPO α=0.3 β=0.1")
    r0_final = aggregate(round0_runs, "Round-0-only DPO β=0.1, SFT after")
    if not (sft_final and hyb_final and r0_final):
        print("aggregate returned None — empty histories?")
        return

    print(f"\n=== Final pass rate comparison ===")
    sft_mu, sft_sd, _ = sft_final
    hyb_mu, hyb_sd, _ = hyb_final
    r0_mu, r0_sd, _ = r0_final
    print(f"  SFT (baseline): {sft_mu:.3f} ± {sft_sd:.3f}")
    print(f"  Hybrid α=0.3 : {hyb_mu:.3f} ± {hyb_sd:.3f}  Δ={hyb_mu-sft_mu:+.3f}")
    print(f"  Round-0-only : {r0_mu:.3f} ± {r0_sd:.3f}  Δ={r0_mu-sft_mu:+.3f}")

    def verdict(arm_mu, arm_sd, name):
        thresh = 2 * max(sft_sd, arm_sd)
        delta = arm_mu - sft_mu
        if abs(delta) > thresh:
            v = "ROBUST WIN" if delta > 0 else "ROBUST LOSS"
        else:
            v = "WITHIN NOISE"
        print(f"  {name:20s} Δ={delta:+.3f}  2σ={thresh:.3f} → {v}")

    print(f"\n=== Significance verdict ===")
    verdict(hyb_mu, hyb_sd, "Hybrid α=0.3")
    verdict(r0_mu, r0_sd, "Round-0-only DPO")

    # Per-round Δ trajectory — look for r=1 spike (Phase 11 S5 K9 1M peak)
    print(f"\n=== Per-round Δ vs SFT trajectory ===")
    n_rounds = len(sft_runs[0]["history"])
    print(f"  {'round':12s}  {'SFT':>6s}  {'Hybrid':>6s}  Δ_h    {'Round0':>6s}  Δ_r0")
    for r_idx in range(n_rounds):
        s = round_stats(sft_runs, r_idx)
        h = round_stats(hybrid_runs, r_idx)
        r = round_stats(round0_runs, r_idx)
        if not (s and h and r):
            continue
        rlabel = sft_runs[0]["history"][r_idx]["label"]
        print(f"  {rlabel:12s}  {s[0]:.3f}  {h[0]:.3f}  {h[0]-s[0]:+.3f}  {r[0]:.3f}  {r[0]-s[0]:+.3f}")


if __name__ == "__main__":
    main()
