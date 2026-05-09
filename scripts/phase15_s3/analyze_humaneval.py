"""Phase 15 S3b analyzer — variance decomposition at HumanEval."""

import json
import math
import statistics
from pathlib import Path

S3_DIR = Path(__file__).parent
S1_DIR = S3_DIR.parent / "phase15_s1"


def final_rate(p):
    return json.loads(Path(p).read_text())["history"][-1]["pass_rate"]


def main():
    init_runs = sorted(S3_DIR.glob("run_he_init*_harvest0.json"))
    harvest_runs = sorted(S3_DIR.glob("run_he_init0_harvest*.json"))
    combined_runs = sorted(S1_DIR.glob("run_seed*.json"))

    init_rates = [final_rate(p) for p in init_runs]
    harvest_rates = [final_rate(p) for p in harvest_runs]
    combined_rates = [final_rate(p) for p in combined_runs]

    def mu_sd(xs):
        if len(xs) < 2:
            return statistics.mean(xs), 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    init_mu, init_sd = mu_sd(init_rates)
    harvest_mu, harvest_sd = mu_sd(harvest_rates)
    combined_mu, combined_sd = mu_sd(combined_rates)

    print(f"\n=== Phase 15 S3b — variance decomposition (HumanEval substrate) ===\n")
    print(f"Axis            n  mean   σ        per-seed")
    print(f"  init-only     {len(init_rates):2d}  {init_mu:.3f}  {init_sd:.3f}    "
          f"{[round(x,3) for x in init_rates]}")
    print(f"  harvest-only  {len(harvest_rates):2d}  {harvest_mu:.3f}  {harvest_sd:.3f}    "
          f"{[round(x,3) for x in harvest_rates]}")
    print(f"  combined (S1) {len(combined_rates):2d}  {combined_mu:.3f}  {combined_sd:.3f}    "
          f"{[round(x,3) for x in combined_rates]}")

    pred = math.sqrt(init_sd**2 + harvest_sd**2)
    print(f"\nAdditivity prediction (independent axes):")
    print(f"  √(σ_init² + σ_harvest²) = √({init_sd:.3f}² + {harvest_sd:.3f}²) = {pred:.3f}")
    print(f"  Observed σ_combined    = {combined_sd:.3f}")
    if combined_sd > 0:
        ratio = pred / combined_sd
        print(f"  Predicted/Observed = {ratio:.2f}")

    print(f"\n=== Decomposition verdict ===")
    init_share = 0.0
    if init_sd > 0 and harvest_sd > 0:
        init_share = (init_sd**2) / (init_sd**2 + harvest_sd**2)
        print(f"  σ_init contributes {init_share*100:.0f}%, σ_harvest contributes {(1-init_share)*100:.0f}%")

    # Compare to Phase 14 S3a
    print(f"\n=== Cross-substrate comparison ===")
    print(f"  Phase 14 substrate (saturated, 25 problems):")
    print(f"    σ_init=0.004 (7%) vs σ_harvest=0.016 (93%)  — harvest-dominated")
    print(f"  Phase 15 substrate (headroom, 164 problems):")
    print(f"    σ_init={init_sd:.3f} ({init_share*100:.0f}%) vs σ_harvest={harvest_sd:.3f} ({(1-init_share)*100:.0f}%)")
    if init_share > 0.6:
        print(f"  → init-dominated at HumanEval; supports S1 mechanism prediction")
    elif init_share < 0.4:
        print(f"  → harvest-dominated at HumanEval; opposite of S1 mechanism prediction")
    else:
        print(f"  → roughly balanced at HumanEval")


if __name__ == "__main__":
    main()
