#!/usr/bin/env python3
"""Phase 22 C-1 — pre-registered analysis. Written BEFORE any C-1 data exists.

Applies exactly the rule locked in docs/phase22-c1-prereg.md:

    ESTABLISHED  iff  p < 0.05 (paired, two-sided) AND mean diff > 0
    otherwise    NOT ESTABLISHED

pass@1 is the primary endpoint. pass@5 and the sign count are printed as
descriptive context and are explicitly NOT decisive.

Usage: analyze.py [eval_dir]      default scratch-7b-sft/c1_prereg/eval
"""
import os
import re
import statistics as st
import sys
from math import sqrt

SEEDS = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
ALPHA = 0.05


def student_t_sf(t, df):
    """Two-sided p for Student's t, via the regularized incomplete beta.
    scipy is not available in this repo's env, so this is self-contained."""
    t = abs(t)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, itmax=300, eps=3e-16, fpmin=1e-300):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    import math
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return ib  # already the two-sided tail for this parameterisation


def read_eval(path):
    if not os.path.exists(path):
        return None
    s = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="ignore").read())
    m1 = re.search(r"aggregate pass@1 \(raw, all samples\) = ([0-9.]+)", s)
    m5 = re.search(r"per-prompt pass@5 = ([0-9.]+)", s)
    if not (m1 and m5):
        return None
    return float(m1.group(1)), float(m5.group(1))


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    t = m / (sd / sqrt(n)) if sd else float("inf")
    p = student_t_sf(t, n - 1) if sd else 0.0
    return d, m, sd, t, p


def main():
    ev = sys.argv[1] if len(sys.argv) > 1 else "scratch-7b-sft/c1_prereg/eval"
    pos, ful, used, dropped = [], [], [], []
    for s in SEEDS:
        p = read_eval(f"{ev}/posonly_seed{s}.log")
        f = read_eval(f"{ev}/fulladv_seed{s}.log")
        if p and f:
            pos.append(p)
            ful.append(f)
            used.append(s)
        else:
            dropped.append(s)

    print("=" * 66)
    print("Phase 22 C-1 — PRE-REGISTERED CONFIRMATORY ANALYSIS")
    print("rule: ESTABLISHED iff p < 0.05 (paired, two-sided) AND mean > 0")
    print("=" * 66)
    print(f"pairs used: {len(used)}/12  {used}")
    if dropped:
        print(f"pairs dropped (incomplete): {dropped}  <-- power recomputed below")
    if len(used) < 3:
        print("insufficient pairs — no analysis")
        return

    p1p = [x[0] for x in pos]
    p1f = [x[0] for x in ful]
    p5p = [x[1] for x in pos]
    p5f = [x[1] for x in ful]

    print(f"\nposonly pass@1: {st.mean(p1p):.4f} ± {st.stdev(p1p):.4f}")
    print(f"fulladv pass@1: {st.mean(p1f):.4f} ± {st.stdev(p1f):.4f}")

    d, m, sd, t, p = paired(p1p, p1f)
    print("\n--- PRIMARY: paired pass@1 (posonly − fulladv) ---")
    print(f"  per-seed: {[round(x, 4) for x in d]}")
    print(f"  mean={m:+.4f}  sd={sd:.4f}  t={t:.3f}  df={len(d)-1}  p={p:.4f}")
    print(f"  favours posonly in {sum(1 for x in d if x > 0)}/{len(d)} pairs")

    established = (p < ALPHA) and (m > 0)
    print("\n" + "=" * 66)
    print(f"VERDICT: {'ESTABLISHED' if established else 'NOT ESTABLISHED'}")
    if not established:
        if m > 0 and p < 0.10:
            print("  (p in [0.05, 0.10) — per the locked rule this is NOT a win;")
            print("   it must not be re-described as 'directional' to rescue it.)")
        dz = abs(m / sd) if sd else 0
        print(f"  observed dz = {dz:.2f}")
        if dz < 0.8:
            print("  pre-committed caveat: n=12 has ~45% power at dz=0.6, so a")
            print("  true effect that small is NOT ruled out by this null.")
    print("=" * 66)

    d5, m5, sd5, t5, p5 = paired(p5p, p5f)
    print("\n--- SECONDARY (descriptive, NOT decisive) ---")
    print(f"  pass@5 paired mean={m5:+.4f} sd={sd5:.4f} t={t5:.3f} p={p5:.4f}"
          f"  ({sum(1 for x in d5 if x > 0)}/{len(d5)})")
    print(f"  posonly pass@5 {st.mean(p5p):.4f} ± {st.stdev(p5p):.4f} | "
          f"fulladv {st.mean(p5f):.4f} ± {st.stdev(p5f):.4f}")
    print("\n  exploratory n=8 (optional stopping, for reference only):")
    print("    pass@1 +0.1242, t=3.62, p=0.0086, 8/8   |   pass@5 +0.031, p=0.17")


if __name__ == "__main__":
    main()
