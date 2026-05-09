"""Phase 13 S2 analyzer: 5-seed × {tiny, medium} K9 results.

Reads /tmp/p13s2_{tiny,medium}.log, extracts per-round metrics per
seed, computes mean ± std, and reports a robustness verdict.

Same logic as scripts/phase13_s1/analyze.py — duplicated to keep
S1's tooling stable while S2 evolves the parser if needed.
"""

import re
import statistics
import sys
from pathlib import Path


def parse_log(log_path):
    text = Path(log_path).read_text()
    blocks = re.split(r"=== (\w+) seed=(\d+) ===", text)
    results = []
    for i in range(1, len(blocks) - 2, 3):
        variant = blocks[i]
        seed = int(blocks[i + 1])
        content = blocks[i + 2]
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


def summarize(results, label):
    sub = [r for r in results if r["variant"] == label]
    if not sub:
        return None
    mean_gen = [r["mean_gen_pass"] for r in sub]
    final_eval = [r["final_eval"] for r in sub]
    best_eval = [r["best_eval"] for r in sub]
    return {
        "variant": label,
        "n": len(sub),
        "mean_gen_avg": statistics.mean(mean_gen),
        "mean_gen_std": statistics.stdev(mean_gen) if len(mean_gen) > 1 else 0.0,
        "mean_gen_per_seed": mean_gen,
        "final_eval_avg": statistics.mean(final_eval),
        "final_eval_std": statistics.stdev(final_eval) if len(final_eval) > 1 else 0.0,
        "final_eval_per_seed": final_eval,
        "best_eval_avg": statistics.mean(best_eval),
        "best_eval_std": statistics.stdev(best_eval) if len(best_eval) > 1 else 0.0,
        "best_eval_per_seed": best_eval,
    }


def fmt(label, mean, std, per_seed):
    s = ", ".join(f"{x:.3f}" if isinstance(x, float) else f"{x}" for x in per_seed)
    return f"  {label:18s} {mean:.3f} ± {std:.3f}  (per-seed: [{s}])"


def main():
    tiny_log = "/tmp/p13s2_tiny.log"
    med_log = "/tmp/p13s2_medium.log"
    results = parse_log(tiny_log) + parse_log(med_log)
    tiny = summarize(results, "tiny")
    medium = summarize(results, "medium")

    if tiny is None or medium is None:
        print("Variants found:", set(r["variant"] for r in results))
        sys.exit(1)

    print(f"\n=== Phase 13 S2 — Scale variance: tiny (1M) vs medium (10M) ===\n")
    print(f"tiny  (~1M, n_layer=4 n_embd=128, n={tiny['n']}):")
    print(fmt("mean_gen_pass", tiny["mean_gen_avg"], tiny["mean_gen_std"], tiny["mean_gen_per_seed"]))
    print(fmt("final_eval/24", tiny["final_eval_avg"], tiny["final_eval_std"], tiny["final_eval_per_seed"]))
    print(fmt("best_eval/24", tiny["best_eval_avg"], tiny["best_eval_std"], tiny["best_eval_per_seed"]))
    print()
    print(f"medium (~10M, n_layer=8 n_embd=256, n={medium['n']}):")
    print(fmt("mean_gen_pass", medium["mean_gen_avg"], medium["mean_gen_std"], medium["mean_gen_per_seed"]))
    print(fmt("final_eval/24", medium["final_eval_avg"], medium["final_eval_std"], medium["final_eval_per_seed"]))
    print(fmt("best_eval/24", medium["best_eval_avg"], medium["best_eval_std"], medium["best_eval_per_seed"]))

    print(f"\n=== Variance comparison ===")
    print(f"  σ(mean_gen)  : tiny {tiny['mean_gen_std']:.4f}  vs  medium {medium['mean_gen_std']:.4f}"
          f"   (medium / tiny = {medium['mean_gen_std'] / max(tiny['mean_gen_std'], 1e-9):.2f}×)")
    print(f"  σ(final_eval): tiny {tiny['final_eval_std']:.3f}   vs  medium {medium['final_eval_std']:.3f}"
          f"    (medium / tiny = {medium['final_eval_std'] / max(tiny['final_eval_std'], 1e-9):.2f}×)")
    print(f"  σ(best_eval) : tiny {tiny['best_eval_std']:.3f}   vs  medium {medium['best_eval_std']:.3f}"
          f"    (medium / tiny = {medium['best_eval_std'] / max(tiny['best_eval_std'], 1e-9):.2f}×)")

    print(f"\n=== Mean comparison (same metric → does scale-up help?) ===")
    print(f"  mean_gen_pass : tiny {tiny['mean_gen_avg']:.3f}  vs  medium {medium['mean_gen_avg']:.3f}"
          f"   (Δ = {medium['mean_gen_avg'] - tiny['mean_gen_avg']:+.3f})")
    print(f"  final_eval/24 : tiny {tiny['final_eval_avg']:.2f}  vs  medium {medium['final_eval_avg']:.2f}"
          f"   (Δ = {medium['final_eval_avg'] - tiny['final_eval_avg']:+.2f})")
    print(f"  best_eval/24  : tiny {tiny['best_eval_avg']:.2f}  vs  medium {medium['best_eval_avg']:.2f}"
          f"   (Δ = {medium['best_eval_avg'] - tiny['best_eval_avg']:+.2f})")


if __name__ == "__main__":
    main()
