"""Phase 9 S4 analyzer: read harvest JSON, compute AUC + selection sweep
in the same shape as critic_baseline_python.rs output."""

import json
import random
import sys
from collections import defaultdict


def roc_auc(scored_labels):
    """Mann–Whitney U formulation. scored_labels: list of (score, label_bool)."""
    pos = [s for s, l in scored_labels if l]
    neg = [s for s, l in scored_labels if not l]
    if not pos or not neg:
        return float("nan")
    n_concord = 0
    n_total = 0
    for sp in pos:
        for sn in neg:
            n_total += 1
            if sp > sn:
                n_concord += 1
            elif sp == sn:
                n_concord += 0.5
    return n_concord / n_total


def bake_off(samples_by_prompt, factor, n_trials, score_key, seed):
    rng = random.Random(seed)
    rand_correct = 0
    crit_correct = 0
    total = 0
    for _ in range(n_trials):
        for cohort in samples_by_prompt.values():
            if len(cohort) < factor:
                continue
            picks = rng.sample(cohort, factor)
            if picks[0]["verdict"]:
                rand_correct += 1
            best = max(picks, key=lambda s: s[score_key])
            if best["verdict"]:
                crit_correct += 1
            total += 1
    if total == 0:
        return float("nan"), float("nan")
    return rand_correct / total, crit_correct / total


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s4_harvest.json"
    d = json.load(open(path))
    model = d["model"]
    samples = d["samples"]
    n = len(samples)
    n_pass = sum(1 for s in samples if s["verdict"])
    pass_rate = n_pass / n if n else 0.0

    print(f"\n=== Phase 9 S4 — Shape C on external HF model ===")
    print(f"model:             {model}")
    print(f"samples:           {n} ({n_pass} correct, {n - n_pass} incorrect)")
    print(f"pass_rate:         {pass_rate:.3f}")

    # Per-challenge breakdown
    by_ch = defaultdict(list)
    for s in samples:
        by_ch[s["challenge"]].append(s)
    print()
    print("per-challenge pass rate:")
    for ch, group in by_ch.items():
        pos = sum(1 for s in group if s["verdict"])
        print(f"  {ch:30s} {pos}/{len(group)}  ({pos/len(group):.1%})")

    # AUC
    pairs_mean = [(s["mean_logp"], s["verdict"]) for s in samples]
    pairs_sum = [(s["sum_logp"], s["verdict"]) for s in samples]
    pairs_random = [(random.Random(0xC417 + i).random(), s["verdict"]) for i, s in enumerate(samples)]
    auc_mean = roc_auc(pairs_mean)
    auc_sum = roc_auc(pairs_sum)
    auc_random = roc_auc(pairs_random)

    print()
    print(f"LogitCritic mean:  {auc_mean:.3f}")
    print(f"LogitCritic sum:   {auc_sum:.3f}   * PRIMARY GATE (Phase 7)")
    print(f"RandomCritic:      {auc_random:.3f}")
    print()
    print("Phase 7 acceptance gate: sum-AUC >= 0.6")
    best = max([x for x in [auc_mean, auc_sum] if x == x], default=float("nan"))
    if best != best:
        print("=> undefined")
    elif best >= 0.6:
        print("=> PASS — Shape C transfers to external HF model.")
    elif best >= 0.5:
        print("=> MARGINAL — calibrated but below deployment gate.")
    elif best >= 0.4:
        print("=> NO SIGNAL.")
    else:
        print("=> ANTI-CALIBRATION — model log-prob anti-correlates with verdict.")

    # Top/bottom by sum logp
    by_sum = sorted(samples, key=lambda s: -s["sum_logp"])
    print("\n=== Top 5 by sum log-prob ===")
    for s in by_sum[:5]:
        ok = "ok" if s["verdict"] else "no"
        print(f"  [{ok}] {s['sum_logp']:.3f}  {s['prompt']!r} -> {s['completion']!r}")
    print("\n=== Bottom 5 by sum log-prob ===")
    for s in by_sum[-5:]:
        ok = "ok" if s["verdict"] else "no"
        print(f"  [{ok}] {s['sum_logp']:.3f}  {s['prompt']!r} -> {s['completion']!r}")

    # Selection sweep
    samples_by_prompt = defaultdict(list)
    for s in samples:
        samples_by_prompt[s["prompt"]].append(s)
    spp = min(len(g) for g in samples_by_prompt.values())
    print(f"\n=== Selection sweep (sum log-prob critic) ===")
    print(f"  F  random_pass  critic_pass    Δ      lift")
    print(f"  -  -----------  -----------  ------   -----")
    for f in [1, 2, 4, 8, 16]:
        if f > spp:
            continue
        rp, cp = bake_off(samples_by_prompt, f, 1000, "sum_logp", 0xB00B + f)
        delta = cp - rp
        lift = cp / rp if rp > 1e-6 else float("inf")
        print(f"  {f}    {rp:>9.3f}    {cp:>9.3f}   {delta:+.3f}   {lift:>4.2f}x")

    print(f"\nReference matrix (in-house):")
    print(f"  K9  RustCode (1M  GPT)  pass=19.0%  AUC=0.727  F=4 lift 1.22x")
    print(f"  P8  Python   (1M  GPT)  pass=35.6%  AUC=0.848  F=4 lift 1.00x")
    print(f"  P9  MultiAss (1M  GPT)  pass= 7.8%  AUC=0.789  F=4 lift 1.32x")


if __name__ == "__main__":
    main()
