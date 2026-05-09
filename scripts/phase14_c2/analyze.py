"""Phase 14 C2 analyzer: compare Muon vs AdamW LoRA training at Qwen
substrate.

Loads:
  scripts/phase14_c2/run_muon_seed*.json   (Muon, 5 seeds)
  scripts/phase14_s1/run_seed*.json        (AdamW baseline, 5 seeds)

Reports:
  - Per-round mean ± σ for both optimizers
  - Final pass rate mean ± σ
  - Δ = mean(Muon) − mean(AdamW)
  - 2σ significance threshold (using max σ of the two arms)
  - Per-challenge focused-subset comparison (the 4 movable problems)

Decision: |Δ| > 2 max(σ_muon, σ_adam) → robust win/loss; else within
noise.
"""

import json
import statistics
from pathlib import Path

C2_DIR = Path(__file__).parent
S1_DIR = C2_DIR.parent / "phase14_s1"


def load_runs(pattern_dir, glob):
    out = []
    for p in sorted(pattern_dir.glob(glob)):
        out.append(json.loads(p.read_text()))
    return out


def aggregate(runs, label):
    n_rounds = max(len(r["history"]) for r in runs)
    print(f"\n--- {label} ({len(runs)} seeds) ---")
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
        rlabel = runs[0]["history"][r_idx]["label"]
        print(f"  {rlabel:10s}  {mu:.3f} ± {sd:.3f}  [{per_seed_s}]")
        if r_idx == n_rounds - 1:
            final_rates = per_seed
    final_mu = statistics.mean(final_rates)
    final_sd = statistics.stdev(final_rates) if len(final_rates) > 1 else 0.0
    return final_mu, final_sd, final_rates


def per_challenge(runs):
    out = {}
    names = list(runs[0]["history"][-1]["per_challenge"].keys())
    for name in names:
        rates = []
        for run in runs:
            ch = run["history"][-1]["per_challenge"].get(name)
            if ch is None or ch["total"] == 0:
                continue
            rates.append(ch["pass"] / ch["total"])
        if rates:
            mu = statistics.mean(rates)
            sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            out[name] = (mu, sd, rates)
    return out


def main():
    muon_runs = load_runs(C2_DIR, "run_muon_seed*.json")
    adam_runs = load_runs(S1_DIR, "run_seed*.json")
    if not muon_runs:
        print(f"no run_muon_seed*.json in {C2_DIR}")
        return
    if not adam_runs:
        print(f"no run_seed*.json in {S1_DIR}")
        return

    print(f"\n=== Phase 14 C2 — Muon vs AdamW for LoRA at Qwen substrate ===")
    print(f"Muon: {len(muon_runs)} seeds, AdamW (S1 baseline): {len(adam_runs)} seeds\n")

    adam_mu, adam_sd, _ = aggregate(adam_runs, "AdamW (S1 baseline)")
    muon_mu, muon_sd, _ = aggregate(muon_runs, "Muon")

    delta = muon_mu - adam_mu
    sigma_max = max(adam_sd, muon_sd)
    threshold = 2 * sigma_max

    print(f"\n=== Final pass rate comparison ===")
    print(f"  AdamW: {adam_mu:.3f} ± {adam_sd:.3f}")
    print(f"  Muon : {muon_mu:.3f} ± {muon_sd:.3f}")
    print(f"  Δ    = {delta:+.3f}  (Muon − AdamW)")
    print(f"  2 max(σ) = {threshold:.3f}  (significance threshold)")
    if abs(delta) > threshold:
        verdict = "ROBUST WIN for Muon" if delta > 0 else "ROBUST LOSS for Muon"
    else:
        verdict = "WITHIN NOISE — no algorithmic signal"
    print(f"  → {verdict}")

    # Focused subset: the non-saturated problems
    print(f"\n=== Per-challenge final pass rate (focused-subset reveal) ===")
    adam_pc = per_challenge(adam_runs)
    muon_pc = per_challenge(muon_runs)
    movable = []
    for name, (a_mu, a_sd, _) in adam_pc.items():
        if a_mu < 0.95 and a_mu >= 0.05:
            movable.append(name)
        elif a_mu < 0.05:
            movable.append(name)  # cold-start: see if Muon can crack
    print(f"  movable (non-saturated under AdamW): {len(movable)}")
    for name in movable:
        a_mu, a_sd, _ = adam_pc[name]
        m_mu, m_sd, _ = muon_pc.get(name, (None, None, None))
        if m_mu is None:
            continue
        d = m_mu - a_mu
        print(f"  {name:30s}  AdamW {a_mu:.3f} ± {a_sd:.3f}  Muon {m_mu:.3f} ± {m_sd:.3f}  Δ={d:+.3f}")

    # Also look at saturated problems — does Muon break any?
    broken_by_muon = []
    for name, (a_mu, _, _) in adam_pc.items():
        if a_mu >= 0.95:
            m_mu, _, _ = muon_pc.get(name, (None, None, None))
            if m_mu is not None and m_mu < 0.95:
                broken_by_muon.append((name, a_mu, m_mu))
    if broken_by_muon:
        print(f"\n  ⚠ Muon broke {len(broken_by_muon)} saturated problem(s):")
        for name, a, m in broken_by_muon:
            print(f"    {name:30s}  AdamW {a:.3f}  Muon {m:.3f}")


if __name__ == "__main__":
    main()
