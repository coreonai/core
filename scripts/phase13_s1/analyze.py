"""Phase 13 S1 A2 analyzer: 5-seed Muon vs AdamW K9 results.

Reads /tmp/p13s1_{adam,muon}_seeds.log, extracts per-round metrics
per seed, computes mean ± std for each variant, and reports a
robustness verdict.

Usage:
    python scripts/phase13_s1/analyze.py
"""

import re
import statistics
import sys
from pathlib import Path


def parse_log(log_path):
    """Return list-of-dicts: one entry per (variant_name, seed).

    Each entry has: variant, seed, round_eval (list of 4),
    round_gen (list of 4), final_eval, mean_gen_pass.
    """
    text = Path(log_path).read_text()
    # Each block looks like:
    #   === <variant> seed=N ===
    #   ...
    #   === history ===
    #   round 0: gen=X/Y (Z%)  eval before=A/B after=C/D  Δ=±E
    #   round 1: ...
    #   round 2: ...
    #   round 3: ...
    #   === <variant> seed=N done ===
    blocks = re.split(r"=== (\w+) seed=(\d+) ===", text)
    results = []
    # blocks alternates: prefix, variant, seed_str, content, variant, seed_str, content, ...
    for i in range(1, len(blocks) - 2, 3):
        variant = blocks[i]
        seed = int(blocks[i + 1])
        content = blocks[i + 2]
        # Stop at next header or end-of-block marker
        match_history = re.search(
            r"=== history ===\n(.*?)(?:===|\Z)", content, re.DOTALL
        )
        if not match_history:
            continue
        hist = match_history.group(1)
        rounds_gen = []
        rounds_eval_after = []
        for m in re.finditer(
            r"round (\d+): gen=(\d+)/(\d+) \(([0-9.]+)%\)\s+"
            r"eval before=(\d+)/(\d+) after=(\d+)/(\d+)\s+Δ=([+-]?\d+)",
            hist,
        ):
            rd = int(m.group(1))
            gen_correct = int(m.group(2))
            gen_total = int(m.group(3))
            eval_after = int(m.group(7))
            while len(rounds_gen) <= rd:
                rounds_gen.append(None)
                rounds_eval_after.append(None)
            rounds_gen[rd] = (gen_correct, gen_total)
            rounds_eval_after[rd] = eval_after
        if not rounds_gen:
            continue
        gen_pass_total = sum(g[0] for g in rounds_gen if g is not None)
        gen_total = sum(g[1] for g in rounds_gen if g is not None)
        results.append({
            "variant": variant,
            "seed": seed,
            "round_eval_after": rounds_eval_after,
            "round_gen": [g[0] for g in rounds_gen if g is not None],
            "mean_gen_pass": (gen_pass_total / gen_total) if gen_total else 0.0,
            "final_eval": rounds_eval_after[-1] if rounds_eval_after else None,
            "best_eval": max((e for e in rounds_eval_after if e is not None), default=0),
        })
    return results


def summarize(results, variant_label):
    sub = [r for r in results if r["variant"] == variant_label]
    if not sub:
        return None
    mean_gen = [r["mean_gen_pass"] for r in sub]
    final_eval = [r["final_eval"] for r in sub]
    best_eval = [r["best_eval"] for r in sub]
    return {
        "variant": variant_label,
        "n_seeds": len(sub),
        "mean_gen_avg": statistics.mean(mean_gen),
        "mean_gen_std": statistics.stdev(mean_gen) if len(mean_gen) > 1 else 0.0,
        "mean_gen_per_seed": mean_gen,
        "final_eval_avg": statistics.mean(final_eval),
        "final_eval_std": statistics.stdev(final_eval) if len(final_eval) > 1 else 0.0,
        "final_eval_per_seed": final_eval,
        "best_eval_avg": statistics.mean(best_eval),
        "best_eval_std": statistics.stdev(best_eval) if len(best_eval) > 1 else 0.0,
    }


def main():
    adam_log = "/tmp/p13s1_adam_seeds.log"
    muon_log = "/tmp/p13s1_muon_seeds.log"
    if not Path(adam_log).exists():
        print(f"missing {adam_log}", file=sys.stderr)
        sys.exit(1)
    if not Path(muon_log).exists():
        print(f"missing {muon_log}", file=sys.stderr)
        sys.exit(1)

    results = parse_log(adam_log) + parse_log(muon_log)
    adam = summarize(results, "AdamW")
    muon = summarize(results, "Muon")

    if adam is None or muon is None:
        print("Could not parse both adam + muon results")
        print("Variants found:", set(r["variant"] for r in results))
        sys.exit(1)

    def fmt_with_std(label, mean, std, per_seed):
        per_seed_s = ", ".join(f"{x:.3f}" if isinstance(x, float) else f"{x}" for x in per_seed)
        return f"  {label:18s} {mean:.3f} ± {std:.3f}  (per-seed: [{per_seed_s}])"

    print(f"\n=== Phase 13 S1 A2 — Muon vs AdamW K9 5-seed variance ===\n")
    print(f"AdamW (n={adam['n_seeds']}):")
    print(fmt_with_std("mean_gen_pass", adam["mean_gen_avg"], adam["mean_gen_std"], adam["mean_gen_per_seed"]))
    print(fmt_with_std("final_eval/24", adam["final_eval_avg"], adam["final_eval_std"], adam["final_eval_per_seed"]))
    print(fmt_with_std("best_eval/24",  adam["best_eval_avg"],  adam["best_eval_std"],  [r["best_eval"] for r in results if r["variant"] == "AdamW"]))
    print()
    print(f"Muon  (n={muon['n_seeds']}):")
    print(fmt_with_std("mean_gen_pass", muon["mean_gen_avg"], muon["mean_gen_std"], muon["mean_gen_per_seed"]))
    print(fmt_with_std("final_eval/24", muon["final_eval_avg"], muon["final_eval_std"], muon["final_eval_per_seed"]))
    print(fmt_with_std("best_eval/24",  muon["best_eval_avg"],  muon["best_eval_std"],  [r["best_eval"] for r in results if r["variant"] == "Muon"]))
    print()

    # Verdicts
    delta_gen = muon["mean_gen_avg"] - adam["mean_gen_avg"]
    pooled_std_gen = ((adam["mean_gen_std"] ** 2 + muon["mean_gen_std"] ** 2) / 2) ** 0.5
    z_gen = delta_gen / pooled_std_gen if pooled_std_gen > 1e-9 else float("inf")

    delta_final = muon["final_eval_avg"] - adam["final_eval_avg"]
    pooled_std_final = ((adam["final_eval_std"] ** 2 + muon["final_eval_std"] ** 2) / 2) ** 0.5
    z_final = delta_final / pooled_std_final if pooled_std_final > 1e-9 else float("inf")

    print("=== Verdicts ===")
    print(f"  Δ mean_gen_pass = Muon − AdamW = {delta_gen:+.3f}  z ≈ {z_gen:+.2f}")
    if abs(z_gen) > 2:
        verdict = "ROBUST" + (" Muon win" if delta_gen > 0 else " AdamW win")
    elif abs(z_gen) > 1:
        verdict = "MARGINAL signal"
    else:
        verdict = "NOISE — within 1σ"
    print(f"    → {verdict}")
    print()
    print(f"  Δ final_eval   = Muon − AdamW = {delta_final:+.3f}  z ≈ {z_final:+.2f}")
    if abs(z_final) > 2:
        verdict = "ROBUST" + (" Muon win" if delta_final > 0 else " AdamW win")
    elif abs(z_final) > 1:
        verdict = "MARGINAL signal"
    else:
        verdict = "NOISE — within 1σ"
    print(f"    → {verdict}")


if __name__ == "__main__":
    main()
