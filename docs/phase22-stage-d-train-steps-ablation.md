# Phase 22 Stage D — train-steps ablation recipe (r=2 regression hypothesis test)

The 5-seed gen-n=164 A batch showed `r=2 < r=1` in 5/5 seeds (mean
Δ(r=2−r=1) = −0.081 ± 0.044), which diverges from Phase 17 S1's
monotonic r=1 < r=2 saturation. Likely cause is over-training:
`--train-steps 100` × 2 rounds = 200 total AdamW steps on a small
(10-27 trajectories) LoRA-rank-16 corpus. Phase 17's Python recipe
may have used fewer steps per round.

This doc captures the **ablation recipe** to run after the current
aggregate eval batch finishes (GPUs free in ~70 min).

## Hypothesis

**H1**: 100 train-steps/round catastrophically over-fits round-0's
chosen corpus, causing round-1 training to override round-0 gains.

**Predicted outcome at train-steps=30**: r=2 ≥ r=1 in ≥3/5 seeds,
mean Δ(r=2−r=1) ≥ −0.02. If the hypothesis is right, the 5-seed
gen-n=32 batch's monotonic behavior (which used train-steps=30)
should reappear at gen-n=164 with the same train-steps.

## Recipe

5 seeds × r=2 × gen-n=164 × **train-steps=30** (everything else
identical to A batch):

```bash
for i in 0 1 2 3 4; do
  gpu=$([ $i -eq 0 ] && echo 5 || ([ $i -eq 1 ] && echo 6 || ([ $i -eq 2 ] && echo 7 || ([ $i -eq 3 ] && echo 1 || echo 0))))
  seedval=$((i*100+100))
  mkdir -p /tmp/phase22d_B_seed${i}
  CUDA_VISIBLE_DEVICES=$gpu ./target/release/examples/phase22_he_mr_sft \
    --seed $seedval --rounds 2 --gen-n 164 --eval-n 32 --eval-passk 3 \
    --train-steps 30 \           # ← only change from A batch
    --max-new-tokens 200 \
    --scratch-dir /tmp/phase22d_B_seed${i}/scratch \
    --out-dir /tmp/phase22d_B_seed${i}/ckpts \
    > /tmp/phase22d_B_seed${i}/run.log 2>&1 &
done
```

Wallclock estimate: same as A batch (~25 min parallel) — train is
not the bottleneck (gen + verify + eval dominate). The only saving
is train-step time but it's ~5s vs ~17s per round → negligible.

## What to look for in the results

Three possible outcomes:

1. **Hypothesis confirmed**: r=2 ≥ r=1 in ≥3/5 seeds, mean Δ(r=2−r=1)
   small/positive. Conclusion: over-training was the culprit;
   train-steps=30 is the right Phase 17 default. Next step: run r=3
   sweep + aggregate eval.

2. **Partial**: r=2 closer to r=1 but still negative in most seeds.
   Conclusion: over-training is part of the story but not all.
   Other hypotheses (corpus distribution, LoRA rank) need
   investigation. Next step: try lr=1e-4 (vs current 2e-4) or
   `--lora-rank 32`.

3. **Regression persists**: r=2 ≪ r=1 again. Conclusion: it's not
   over-training; some other Pekko-side divergence from Phase 17's
   Python recipe. Next step: byte-compare the SFT corpus rendering
   (CuratorActor → trainer text format) between Pekko and Phase 17
   Python; the difference is likely in chosen-trajectory formatting.

## G1 results (per-round eval, eval-n=32 × passk=3)

**Hypothesis CONFIRMED** — outcome #1.

```
seed=100  round 0  gen=13/164  pass@3=0.094→0.125  Δ=+0.031   round 1  gen=22/164  0.125→0.125  Δ=+0.000
seed=200  round 0  gen=17/164  pass@3=0.156→0.125  Δ=-0.031   round 1  gen=11/164  0.125→0.250  Δ=+0.125
seed=300  round 0  gen=12/164  pass@3=0.188→0.250  Δ=+0.062   round 1  gen=28/164  0.250→0.219  Δ=-0.031
seed=400  round 0  gen=13/164  pass@3=0.281→0.375  Δ=+0.094   round 1  gen=18/164  0.375→0.406  Δ=+0.031
seed=500  round 0  gen=10/164  pass@3=0.156→0.312  Δ=+0.156   round 1  gen=25/164  0.312→0.344  Δ=+0.031
```

| metric | A-batch (steps=100) | **G1 (steps=30)** |
|---|---|---|
| mean r=1 pass@3 | 0.331 | 0.237 |
| mean r=2 pass@3 | 0.250 | **0.269** |
| mean Δ(r=2−r=1) | **−0.081** | **+0.031** |
| seeds r=2 ≥ r=1 | 0/5 | **4/5** |
| mean Δ(r=2−base) | +0.075 | +0.094 |

**Reading**:
- A-batch had 5/5 r=2 < r=1 (catastrophic). G1 has 4/5 r=2 ≥ r=1
  (monotonic — seed 300 is the only −0.031, within noise).
- Per-round r=1 is LOWER at train-steps=30 (less training, less
  immediate lift) but compounding is now **preserved**.
- Phase 17 mechanism (multi-round SFT compounds positively) requires
  train-steps ≈ 30 at gen-n=164 + rank-16 LoRA + lr=2e-4.

Hypothesis #1 (over-training) CONFIRMED.

**Aggregate eval (Phase 17 metric)** — running on freed GPUs after
G1 training finished. Will compare to A-batch's r=2 = 0.116 (HALF
base) to see if G1's r=2 aggregate recovers to ≥ base = 0.222.

## What this ablation does NOT do

- **Aggregate eval (Phase 17 metric)**: this run uses per-round
  eval-n=32×passk=3 for cheap directional signal. The
  Phase-17-aligned aggregate measurement is a separate
  `phase22_humaneval_baseline --sequential --aggregate --checkpoint
  r=2_merged.safetensors` pass per seed.
- **rank=32 LoRA test**: held for the partial-outcome case.
- **MBPP cross-substrate test**: cross-substrate ablation is a
  follow-up after we know whether HumanEval Stage D mechanism is
  fixed.

## Related work

- A batch: `docs/phase22-stage-d-A-batch-gen-n-164.md` — the run
  that surfaced the regression.
- A-batch aggregate eval: ~70 min wallclock remaining at time of
  this doc. Once landed, the 5 Phase-17-aligned r=2 numbers
  (`phase22_humaneval_baseline --sequential --aggregate` on
  `r0_merged.r1.safetensors`) will tell us how bad the regression
  really is on the apples-to-apples metric.
- Memory: `phase22_stage_d_r2_regression.md` — finding writeup.
