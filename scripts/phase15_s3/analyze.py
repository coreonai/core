"""Phase 15 S3a analyzer: variance decomposition of Phase 14 substrate.

Loads:
  scripts/phase15_s3/run_init*_harvest0.json    (init-axis: 5 inits, harvest=0)
  scripts/phase15_s3/run_init0_harvest*.json    (harvest-axis: init=0, 5 harvests)
  scripts/phase14_s1/run_seed*.json             (combined: paired seed=init=harvest)

Reports σ_init, σ_harvest, σ_combined and checks the additivity
prediction σ_combined² ≈ σ_init² + σ_harvest². If σ_combined is much
larger than predicted, there's interaction (or other entangled axes).
"""

import json
import math
import statistics
from pathlib import Path

S3_DIR = Path(__file__).parent
S1_DIR = S3_DIR.parent / "phase14_s1"


def load_json(p):
    return json.loads(Path(p).read_text())


def final_rate(run_json):
    return run_json["history"][-1]["pass_rate"]


def main():
    init_runs = sorted(S3_DIR.glob("run_init*_harvest0.json"))
    harvest_runs = sorted(S3_DIR.glob("run_init0_harvest*.json"))
    combined_runs = sorted(S1_DIR.glob("run_seed*.json"))

    if not (init_runs and harvest_runs and combined_runs):
        print("missing one of: init / harvest / combined runs")
        print(f"  init={len(init_runs)} harvest={len(harvest_runs)} combined={len(combined_runs)}")
        return

    init_rates = [final_rate(load_json(p)) for p in init_runs]
    harvest_rates = [final_rate(load_json(p)) for p in harvest_runs]
    combined_rates = [final_rate(load_json(p)) for p in combined_runs]

    def mu_sd(xs):
        if len(xs) < 2:
            return statistics.mean(xs), 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    init_mu, init_sd = mu_sd(init_rates)
    harvest_mu, harvest_sd = mu_sd(harvest_rates)
    combined_mu, combined_sd = mu_sd(combined_rates)

    print(f"\n=== Phase 15 S3a — variance decomposition (Phase 14 substrate) ===\n")
    print(f"Axis            n  mean   σ        per-seed")
    print(f"  init-only     {len(init_rates):2d}  {init_mu:.3f}  {init_sd:.3f}    "
          f"{[round(x,3) for x in init_rates]}")
    print(f"  harvest-only  {len(harvest_rates):2d}  {harvest_mu:.3f}  {harvest_sd:.3f}    "
          f"{[round(x,3) for x in harvest_rates]}")
    print(f"  combined (S1) {len(combined_rates):2d}  {combined_mu:.3f}  {combined_sd:.3f}    "
          f"{[round(x,3) for x in combined_rates]}")

    # Additivity check: if init/harvest are independent, σ_total ≈ √(σ_init² + σ_harvest²)
    pred = math.sqrt(init_sd**2 + harvest_sd**2)
    print(f"\nAdditivity prediction (independent axes):")
    print(f"  √(σ_init² + σ_harvest²) = √({init_sd:.3f}² + {harvest_sd:.3f}²) = {pred:.3f}")
    print(f"  Observed σ_combined    = {combined_sd:.3f}")
    if combined_sd > 0:
        ratio = pred / combined_sd
        print(f"  Predicted/Observed = {ratio:.2f}")
        if 0.7 < ratio < 1.3:
            print(f"  → axes appear INDEPENDENT (within ±30%)")
        elif ratio > 1.3:
            print(f"  → predicted > observed — measurement entanglement makes σ_combined LOWER than independent sum")
        else:
            print(f"  → predicted < observed — interaction or hidden axis inflates combined σ")

    print(f"\n=== Decomposition verdict ===")
    if init_sd > 0 and harvest_sd > 0:
        init_share = (init_sd**2) / (init_sd**2 + harvest_sd**2)
        print(f"  σ_init contributes {init_share*100:.0f}%, σ_harvest contributes {(1-init_share)*100:.0f}%")
    print(f"\nPhase 14 σ=0.011 framing: {'mostly LoRA-init noise' if init_sd > harvest_sd else 'mostly sampling noise' if harvest_sd > init_sd else 'split evenly'}")


if __name__ == "__main__":
    main()
