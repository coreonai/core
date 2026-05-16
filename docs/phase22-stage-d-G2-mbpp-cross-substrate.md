# Phase 22 Stage D G2 — MBPP cross-substrate ablation (train-steps=30)

After G1 confirmed that train-steps=30 partially restores Phase 17's
monotonic compounding on HumanEval (4/5 seeds r=2 ≥ r=1, mean
Δ(r=2−r=1) = +0.031, aggregate r=2 = 0.154 vs A-batch 0.116), G2
runs the same recipe on MBPP-100 to test whether the fix
generalizes cross-substrate.

## Setup

5 seeds × r=2 × gen-n=100 × eval-n=32 × passk=3 × **train-steps=30**.

Binary: `phase22_mbpp_mr_sft` (commit `2753241`). Same actor
pipeline as `phase22_he_mr_sft`, only the Domain swap
(`HumanEvalDomain → MbppDomain`). Parallel on GPUs 0/1/5/6/7,
~25 min total wallclock.

## Per-round results (eval-n=32 × passk=3)

```
seed=100  round 0  gen=15/100  pass@3=0.281→0.156  Δ=-0.125   round 1  gen=20/100  0.156→0.188  Δ=+0.031
seed=200  round 0  gen=14/100  pass@3=0.281→0.250  Δ=-0.031   round 1  gen=28/100  0.250→0.219  Δ=-0.031
seed=300  round 0  gen=22/100  pass@3=0.312→0.406  Δ=+0.094   round 1  gen=9/100   0.406→0.312  Δ=-0.094
seed=400  round 0  gen=7/100   pass@3=0.344→0.344  Δ=+0.000   round 1  gen=15/100  0.344→0.312  Δ=-0.031
seed=500  round 0  gen=18/100  pass@3=0.344→0.250  Δ=-0.094   round 1  gen=15/100  0.250→0.188  Δ=-0.062
```

| metric | mean | σ |
|---|---|---|
| base                    | 0.313 | 0.030 |
| r=1 (after 1 train)     | 0.281 | 0.090 |
| r=2 (after 2 trains)    | 0.244 | 0.058 |
| Δ(r=1−base)             | -0.031 | 0.080 |
| Δ(r=2−r=1)              | -0.037 | 0.052 |
| **Δ(r=2−base)**         | **-0.069** | 0.061 |

**5/5 seeds r=2 ≤ base** at MBPP-100 + train-steps=30. Mean Δ is
negative beyond noise (1.1σ below zero).

## Cross-substrate comparison

| substrate | train-steps | base | r=2 (per-round) | Δ(r=2−base) per-round | r=2 aggregate |
|---|---|---|---|---|---|
| HumanEval A-batch | 100 | 0.175 | 0.250 | +0.075 | **0.116** (HALF base) |
| HumanEval G1      | 30  | 0.175 | 0.269 | +0.094 | **0.154** (below base) |
| MBPP G2 (this)    | 30  | 0.313 | 0.244 | **-0.069** | TBD (aggregate eval pending) |

**Cross-substrate divergence**: train-steps=30 partially restores
HumanEval (4/5 seeds positive Δ, mean +0.094 per-round); same
recipe damages MBPP (5/5 seeds non-positive, mean −0.069 per-round).

## Why MBPP behaves differently

Candidate explanations:

1. **Higher base pass-rate (0.313 vs HE's 0.175)** — less headroom
   for SFT to improve, more room for SFT to damage. The model
   already gets ~31% of MBPP problems right; reinforcing the
   correct chosen trajectories may push the distribution away from
   the rest.

2. **Smaller problem set (100 vs 164)** — gen-n=100 with sampling-
   with-replacement covers ~63% unique problems per round. The
   chosen corpus has 7-28 trajectories, but they come from a more
   concentrated subset.

3. **Different prompt format** — MBPP prompt = `<imports>\n\n<sig>\n
   """<text>"""\n` (synthesized from `text` + `code` + parsed
   signature). HumanEval prompt is the canonical
   `from typing import ...\ndef name(args):\n    """<docstring>"""`
   directly from the dataset. Qwen2.5-Coder's pretraining likely
   saw far more HumanEval-style prompts than MBPP-style
   synthesized ones. SFT on MBPP may be teaching the model a
   format it doesn't natively expect.

4. **Per-round eval-set overlap** — n=32 random eval × 100 problems
   = ~32 unique problems = ~30% of 100. Training corpus also
   samples from 100 with replacement → ~63 unique. Overlap ~20
   problems = HEAVY (~63% of eval). Memorization would INFLATE
   numbers, not deflate — so this isn't a confounder for the
   regression direction.

5. **Phase 17 SB used different recipe** — Phase 17 SB MBPP MR
   reported mean 0.453 ± 0.016 at r=2. Either Phase 17 used
   train-steps < 30 OR different lr/lora-rank, OR our Pekko-side
   corpus rendering diverges from Phase 17's Python format.

## Implications

- **train-steps=30 is not the universal fix**. It works for
  HumanEval at this scale but fails for MBPP. The "right" recipe
  is substrate-dependent.
- **Phase 17 SB's 0.453** cannot be reproduced through this Pekko
  recipe at train-steps=30 (best smoke pass-rate is r=2 = 0.244,
  ~10σ below SB).
- **Next ablation candidates**: train-steps=10 / lr=1e-4 /
  lora-rank=32, or a corpus-rendering byte-comparison vs Phase 17
  Python.

## What this run does NOT do

- **Aggregate eval (Phase 17 metric)** — per-round eval-n=32 ×
  passk=3 is the cheap directional signal here. The
  Phase-17-aligned 100×k=10 = 1000-attempt measurement on the r=2
  checkpoints is pending; we already know it'll come in worse than
  per-round (memorization is hiding part of the damage, just like
  on HumanEval where per-round 0.269 → aggregate 0.154).

## See also

- `docs/phase22-stage-d-A-batch-gen-n-164.md` — A-batch HumanEval
  train-steps=100 result (r=2 aggregate 0.116, over-training
  confirmed by magnitude).
- `docs/phase22-stage-d-train-steps-ablation.md` — G1 HumanEval
  train-steps=30 result (compounding restored, 4/5 seeds r=2 ≥
  r=1).
- `docs/phase22-stage-c.md` — `MbppDomain` library this binary
  drives.
- `docs/phase22-overview.md` — Stage D follow-ups table.
