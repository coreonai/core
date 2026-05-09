"""Phase 15 S2 analyzer: OPD multi-teacher student vs SFT-on-union
baseline (Phase 15 S1 reused).

Decision gate: |Δ_final| > 2σ_max(S1, S2-OPD) → robust win/loss.
"""

import json
import statistics
from pathlib import Path

S2_DIR = Path(__file__).parent
S1_DIR = S2_DIR.parent / "phase15_s1"


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
        print(f"  {rlabel:10s}  {mu:.3f} ± {sd:.3f}  [{rs}]")
        if r_idx == n_rounds - 1:
            final = (mu, sd, rates)
    return final


def main():
    sft_runs = load_runs(S1_DIR, "run_seed*.json")
    opd_runs = load_runs(S2_DIR, "run_opd_seed*.json")
    if not (sft_runs and opd_runs):
        print("missing one of: SFT (S1) / OPD (S2) run files")
        return

    print(f"\n=== Phase 15 S2 — Multi-teacher OPD vs SFT-union at HumanEval ===")
    print(f"SFT (S1): {len(sft_runs)} seeds   OPD: {len(opd_runs)} seeds\n")

    sft_final = aggregate(sft_runs, "SFT-union (Phase 15 S1)")
    opd_final = aggregate(opd_runs, "OPD multi-teacher")
    if not (sft_final and opd_final):
        print("aggregate returned None")
        return

    sft_mu, sft_sd, _ = sft_final
    opd_mu, opd_sd, _ = opd_final
    delta = opd_mu - sft_mu
    sigma_max = max(sft_sd, opd_sd)
    threshold = 2 * sigma_max

    print(f"\n=== Final pass rate comparison ===")
    print(f"  SFT-union: {sft_mu:.3f} ± {sft_sd:.3f}")
    print(f"  OPD      : {opd_mu:.3f} ± {opd_sd:.3f}")
    print(f"  Δ        = {delta:+.3f}  (OPD − SFT)")
    print(f"  2σ_max   = {threshold:.3f}")
    if abs(delta) > threshold:
        v = "ROBUST WIN for OPD (mean shift)" if delta > 0 else "ROBUST LOSS for OPD"
    else:
        v = "WITHIN NOISE on mean"
    print(f"  → {v}")

    # S1 mechanism analysis (4-seed) showed FLAT seeds overfit (low train
    # loss, low gen). OPD's KL-anchor is a natural regularizer. So OPD
    # may win on VARIANCE reduction even when mean is similar.
    if sft_sd > 0:
        sd_ratio = opd_sd / sft_sd
        print(f"\n=== Variance comparison (overfitting regularization signal) ===")
        print(f"  σ_OPD / σ_SFT = {sd_ratio:.2f}")
        if sd_ratio < 0.5:
            print(f"  → OPD substantially reduces variance ({1/sd_ratio:.1f}× tighter)")
            print(f"  → Likely regularization win even if mean is flat — see per-seed lifts")
        elif sd_ratio < 0.85:
            print(f"  → OPD modestly reduces variance")
        elif sd_ratio > 1.5:
            print(f"  → OPD INCREASES variance — destabilizing (Phase 14 C3 hybrid pattern)")

    # Specialist meta dump if present
    print("\n=== Specialist metadata ===")
    spec_dir = Path(__file__).parent.parent.parent / "checkpoints" / "phase15_s2"
    for sub in ("strings", "numbers", "collections"):
        meta = spec_dir / f"specialist_{sub}" / "meta.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            final_h = d["history"][-1]
            print(f"  {sub:12s}  n={d['n_challenges']:3d}  final pass="
                  f"{final_h['pass_rate']:.3f} ({final_h['n_pass']}/{final_h['n']})")
        else:
            print(f"  {sub:12s}  (no meta.json yet)")

    # Per-subset student behavior (mirror of S1 mechanism analysis)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from routing import SUBSETS, classify  # noqa: E402

    def per_subset_rate(runs):
        out = {sub: [] for sub in SUBSETS}
        for run in runs:
            per_ch = run["history"][-1]["per_challenge"]
            sub_pass = {sub: [0, 0] for sub in SUBSETS}
            for ch_name, stats in per_ch.items():
                # Look up the prompt from S1 challenges to classify
                # (we don't store prompt in run JSON; use ch_name lookup
                # against the challenge list)
                for ch in [c for items in SUBSETS.values() for c in items]:
                    if ch["name"] == ch_name:
                        sub = classify(ch["prompt"])
                        sub_pass[sub][0] += stats["pass"]
                        sub_pass[sub][1] += stats["total"]
                        break
            for sub in SUBSETS:
                if sub_pass[sub][1]:
                    out[sub].append(sub_pass[sub][0] / sub_pass[sub][1])
        return out

    print("\n=== Per-subset final pass rate (student vs teacher) ===")
    sft_per_sub = per_subset_rate(sft_runs)
    opd_per_sub = per_subset_rate(opd_runs)
    print(f"  {'subset':12s}  {'SFT mean':>8s}  {'OPD mean':>8s}  {'Δ':>6s}  {'spec.':>6s}")
    for sub in ("strings", "numbers", "collections"):
        sft_rate = statistics.mean(sft_per_sub[sub]) if sft_per_sub[sub] else 0
        opd_rate = statistics.mean(opd_per_sub[sub]) if opd_per_sub[sub] else 0
        delta = opd_rate - sft_rate
        spec_path = spec_dir / f"specialist_{sub}" / "meta.json"
        spec_rate = 0
        if spec_path.exists():
            spec_meta = json.loads(spec_path.read_text())
            spec_rate = spec_meta["history"][-1]["pass_rate"]
        print(f"  {sub:12s}  {sft_rate:.3f}    {opd_rate:.3f}    {delta:+.3f}  {spec_rate:.3f}")
    print("  (specialist column = teacher's own final pass on its subset)")
    print("  Hint: if OPD > SFT only on subsets where specialist > SFT_baseline → distillation working")


if __name__ == "__main__":
    main()
