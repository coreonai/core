"""Phase 13 S3 analyzer.

Two questions:
  (a) S3-isolate: at fixed 3-challenge K9, does going 1M → 10M
      reduce variance? (Tests Stage B 'σ shrinks with scale')
  (b) S3-budget: at 10-challenge K9, does giving 10M model 5K
      pretrain (vs S2's 1500) close the gap to S1's 3-challenge
      AdamW result (5.6 ± 3.4 final eval)? (Tests whether the
      A1 confound is curable with more compute.)

Compares against:
  - Phase 13 S1 AdamW (3-challenge tiny) baseline
  - Phase 13 S2 medium (10-challenge, 1500 pretrain) baseline
"""

import re
import statistics
from pathlib import Path


def parse_log(log_path):
    text = Path(log_path).read_text()
    blocks = re.split(r"=== (\w+) seed=(\d+)(?:\s+gpu=\d+)? ===", text)
    results = []
    for i in range(1, len(blocks) - 2, 3):
        variant = blocks[i]
        seed = int(blocks[i + 1])
        content = blocks[i + 2]
        m = re.search(r"=== history ===\n(.*?)(?:===|\Z)", content, re.DOTALL)
        if not m:
            continue
        hist = m.group(1)
        rounds_gen, rounds_eval_after = [], []
        for r in re.finditer(
            r"round (\d+): gen=(\d+)/(\d+) \(([0-9.]+)%\)\s+"
            r"eval before=(\d+)/(\d+) after=(\d+)/(\d+)\s+Δ=([+-]?\d+)",
            hist,
        ):
            rd = int(r.group(1))
            while len(rounds_gen) <= rd:
                rounds_gen.append(None)
                rounds_eval_after.append(None)
            rounds_gen[rd] = (int(r.group(2)), int(r.group(3)))
            rounds_eval_after[rd] = int(r.group(7))
        if not rounds_gen:
            continue
        gp = sum(g[0] for g in rounds_gen if g)
        gt = sum(g[1] for g in rounds_gen if g)
        results.append({
            "variant": variant, "seed": seed,
            "round_eval_after": rounds_eval_after,
            "mean_gen_pass": gp / gt if gt else 0.0,
            "final_eval": rounds_eval_after[-1] if rounds_eval_after else 0,
            "best_eval": max((e for e in rounds_eval_after if e is not None), default=0),
        })
    return results


def summarize(results, label):
    sub = [r for r in results if r["variant"] == label]
    if not sub:
        return None
    mg = [r["mean_gen_pass"] for r in sub]
    fe = [r["final_eval"] for r in sub]
    be = [r["best_eval"] for r in sub]
    return {
        "label": label, "n": len(sub),
        "mean_gen_avg": statistics.mean(mg),
        "mean_gen_std": statistics.stdev(mg) if len(mg) > 1 else 0.0,
        "mean_gen_per_seed": mg,
        "final_eval_avg": statistics.mean(fe),
        "final_eval_std": statistics.stdev(fe) if len(fe) > 1 else 0.0,
        "final_eval_per_seed": fe,
        "best_eval_avg": statistics.mean(be),
        "best_eval_std": statistics.stdev(be) if len(be) > 1 else 0.0,
        "best_eval_per_seed": be,
    }


def fmt(label, mean, std, per_seed):
    s = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in per_seed)
    return f"  {label:18s} {mean:.3f} ± {std:.3f}  (per-seed: [{s}])"


def show(s):
    if s is None:
        return
    print(fmt("mean_gen_pass", s["mean_gen_avg"], s["mean_gen_std"], s["mean_gen_per_seed"]))
    print(fmt("final_eval/24", s["final_eval_avg"], s["final_eval_std"], s["final_eval_per_seed"]))
    print(fmt("best_eval/24", s["best_eval_avg"], s["best_eval_std"], s["best_eval_per_seed"]))


def main():
    s3a = parse_log("/tmp/p13s3a_isolate_tiny.log") + parse_log("/tmp/p13s3a_isolate_medium.log")
    s3b = parse_log("/tmp/p13s3b_budget_medium_gpu2.log") + parse_log("/tmp/p13s3b_budget_medium_gpu3.log")

    isolate_tiny = summarize(s3a, "isolate_tiny")
    isolate_medium = summarize(s3a, "isolate_medium")
    budget_medium = summarize(s3b, "budget_medium")

    print(f"\n=== Phase 13 S3 (a) — isolate scale at 3-challenge K9 ===\n")
    print("isolate_tiny  (~1M, 3-challenge, 1500 pretrain):")
    show(isolate_tiny)
    print()
    print("isolate_medium (~10M, 3-challenge, 1500 pretrain):")
    show(isolate_medium)
    print()
    if isolate_tiny and isolate_medium:
        print(f"  Δ best_eval (medium − tiny) = {isolate_medium['best_eval_avg'] - isolate_tiny['best_eval_avg']:+.2f}")
        print(f"  σ ratio (medium/tiny):")
        print(f"    σ(mean_gen)  : {isolate_medium['mean_gen_std'] / max(isolate_tiny['mean_gen_std'], 1e-9):.2f}×")
        print(f"    σ(final_eval): {isolate_medium['final_eval_std'] / max(isolate_tiny['final_eval_std'], 1e-9):.2f}×")
        print(f"    σ(best_eval) : {isolate_medium['best_eval_std'] / max(isolate_tiny['best_eval_std'], 1e-9):.2f}×")

    print(f"\n=== Phase 13 S3 (b) — budget medium at 10-challenge K9 ===\n")
    print("budget_medium (~10M, 10-challenge, **5000 pretrain**):")
    show(budget_medium)
    print()
    if budget_medium:
        # Compare to Phase 13 S2 medium (10-ch, 1500 pretrain): final 0.2 ± 0.45, best 3.6 ± 2.9
        print(f"  Δ vs S2 medium (10-ch, 1500 pretrain):")
        print(f"    final_eval: {budget_medium['final_eval_avg'] - 0.2:+.2f}  (S2 was 0.2 ± 0.45)")
        print(f"    best_eval : {budget_medium['best_eval_avg'] - 3.6:+.2f}  (S2 was 3.6 ± 2.9)")
        print(f"  Δ vs S1 tiny (3-ch, 1500 pretrain) baseline:")
        print(f"    final_eval: {budget_medium['final_eval_avg'] - 5.6:+.2f}  (S1 was 5.6 ± 3.4)")
        print(f"    best_eval : {budget_medium['best_eval_avg'] - 9.8:+.2f}  (S1 was 9.8 ± 1.6)")


if __name__ == "__main__":
    main()
