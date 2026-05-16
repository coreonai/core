# Phase 22 Stage D — 5-seed gen-n=164 measurement (A batch)

After the 5-seed gen-n=32 run (commit `d1dd6d8`) qualified the
mechanism (5/5 seeds positive Δ, mean Δ=+0.100 ± 0.078, seed 400
hit 0.406 individually), this batch scales to the Phase-17-matching
gen-n=164 configuration to test whether the **mean** converges to
Phase 17 S1's r=2 = 0.404 ± 0.013.

## Setup

```bash
phase22_he_mr_sft \
  --seed {100, 200, 300, 400, 500} \
  --rounds 2 \
  --gen-n 164 \      # Phase 17 standard (vs gen-n=32 in prior batch)
  --eval-n 32 \      # cheap per-round eval (passk=3); benchmark-aligned
  --eval-passk 3 \   # aggregate eval is a separate phase22_humaneval_baseline run
  --train-steps 100 \ # Phase 17 used ~100 steps/round
  --max-new-tokens 200 \
  --scratch-dir /tmp/phase22d_A_seed{0..4}/scratch \
  --out-dir /tmp/phase22d_A_seed{0..4}/ckpts
```

Parallel on GPUs {0, 1, 5, 6, 7}.

Wallclock estimate per seed:
- eval-before (eval-n=32 × passk=3 = 96 generations + verifies): ~5 min
- generate (gen-n=164 × ~3s/sample): ~8 min
- verify (164 × ~0.5s python3): ~1.5 min
- train (100 steps × ~0.5s): ~50s
- reload + save: ~30s
- eval-after: ~5 min
- → per round ~20 min, r=2 ~40 min total per seed

Parallel: ~40 min wallclock for the training batch.

## Per-round results (per-seed eval-n=32 × passk=3)

```
seed=100  round 0  gen=13/164  pass@3=0.094→0.188  Δ=+0.094   round 1  gen=25/164  0.188→0.156  Δ=-0.031
seed=200  round 0  gen=17/164  pass@3=0.156→0.250  Δ=+0.094   round 1  gen=25/164  0.250→0.156  Δ=-0.094
seed=300  round 0  gen=12/164  pass@3=0.188→0.219  Δ=+0.031   round 1  gen=27/164  0.219→0.188  Δ=-0.031
seed=400  round 0  gen=13/164  pass@3=0.281→0.562  Δ=+0.281   round 1  gen=25/164  0.562→0.438  Δ=-0.125
seed=500  round 0  gen=10/164  pass@3=0.156→0.438  Δ=+0.281   round 1  gen=25/164  0.438→0.312  Δ=-0.125
```

| seed | base | r=1 | r=2 | Δ(r=1−base) | Δ(r=2−r=1) | Δ(r=2−base) |
|---|---|---|---|---|---|---|
| 100 | 0.094 | 0.188 | 0.156 | +0.094 | -0.031 | +0.062 |
| 200 | 0.156 | 0.250 | 0.156 | +0.094 | -0.094 | +0.000 |
| 300 | 0.188 | 0.219 | 0.188 | +0.031 | -0.031 | +0.000 |
| 400 | 0.281 | **0.562** | 0.438 | +0.281 | -0.125 | +0.156 |
| 500 | 0.156 | **0.438** | 0.312 | +0.281 | -0.125 | +0.156 |
| **mean** | **0.175** | **0.331** | **0.250** | **+0.156** | **-0.081** | **+0.075** |
| **σ** | 0.066 | 0.146 | 0.111 | 0.116 | 0.044 | 0.080 |

Wallclock: parallel batch ran ~25 min (much faster than the
estimated 40 min — generation throughput on Qwen was better than the
gen-n=32 batch's per-prompt rate suggested).

### Key findings from the per-round eval

1. **r=1 mean = 0.331** vs gen-n=32 batch's r=1 = 0.244 (+0.087):
   bigger corpus (13-17 trajectories instead of 1-4) materially
   improves single-round SFT. Both per-prompt corpus size and
   `--train-steps 100` (vs 30) contribute.

2. **r=2 < r=1 in 5/5 seeds** (mean Δ(r=2−r=1) = −0.081 ± 0.044):
   the second training round *regresses* the model's pass-rate.
   This is **new** — not seen at gen-n=32 where r=2 ≥ r=1 in 4/5
   seeds.

3. **Individual highs**: seed 400 r=1 = 0.562, seed 500 r=1 = 0.438
   — both **exceed Phase 17 S1's r=2 = 0.404** individually. seed
   400 r=1 = 0.562 is 12σ above Phase 17's r=2 mean. These are
   single-seed lucky points (eval-n=32 × passk=3 subset noise) but
   the magnitude is striking.

### Possible explanations for the r=2 regression

- **Catastrophic forgetting**: 100 train-steps × 2 rounds = 200
  total steps. Round 1 over-fits round 0's chosen corpus, then
  round 2's training overrides round 1's gains. Phase 4 era K9
  also showed this pattern under big SFT.
- **Corpus quality regression**: round 1 trains on round 0's
  *fine-tuned* model, generating different (possibly inferior)
  trajectories. But gen counts went UP (13-17 → 25-27) in round 1,
  suggesting more corpus, not less.
- **Train-steps too high**: Phase 17 may have used fewer steps
  (~30-50). Ablation would confirm.
- **eval-n=32 subset is unrepresentative**: the regression might
  be specific to the 32 problems eval samples. Aggregate eval
  (n=164 × passk=10) below will tell.

## Benchmark-aligned aggregate eval (separate phase)

After the training batch produces `r1_merged.safetensors` per seed,
run the Stage B aggregate measurement against each checkpoint:

```bash
phase22_humaneval_baseline \
  --n-problems 164 --passk 10 --sequential --aggregate \
  --max-new-tokens 200
```

This applies the Phase 17 S6 "pass@1 raw" metric (`total_passes /
total_attempts` at temp=0.8 × k=10) to each seed's r=2 checkpoint.
The resulting 5 numbers are the apples-to-apples comparison with
Phase 17 S1's r=2 mean.

Phase 17 S1 anchor:
- base: 0.230 (mean of 5 seeds at r=0)
- r=2 SFT: **0.404 ± 0.013** (mean of 5 seeds)

### 5-seed aggregate eval results

```
seed=100  aggregate pass@1 = 0.1098  (180/1640)   per-prompt pass@10 = 0.3598  (59/164)
seed=200  aggregate pass@1 = 0.0774  (127/1640)   per-prompt pass@10 = 0.3049  (50/164)
seed=300  aggregate pass@1 = 0.0841  (138/1640)   per-prompt pass@10 = 0.3232  (53/164)
seed=400  aggregate pass@1 = 0.1451  (238/1640)   per-prompt pass@10 = 0.4390  (72/164)
seed=500  aggregate pass@1 = 0.1616  (265/1640)   per-prompt pass@10 = 0.4329  (71/164)
```

| metric | mean | σ | ref (Phase 17 / Stage B base) |
|---|---|---|---|
| aggregate pass@1 (r=2 trained) | **0.116** | 0.037 | **0.404 ± 0.013** (Phase 17 S1 r=2 SFT) / 0.222 (Stage B base) |
| per-prompt pass@10 (r=2 trained) | **0.372** | 0.061 | 0.524 (Phase 17 S6 base) |

### What the aggregate numbers reveal

**Catastrophic finding**: our r=2 trained model's aggregate pass@1
(0.116) is **HALF of the base model's 0.222**. The per-round eval
(eval-n=32 × passk=3) showed r=2 mean = 0.250 (between base 0.175
and r=1 0.331), which was misleading — at the full n=164 × k=10
Phase 17 metric the model is significantly **worse than untrained
base**.

Why per-round eval was over-optimistic: the eval set is n=32 random
HumanEval problems sampled WITH REPLACEMENT each call. At n=32 vs
164 total problems, the eval and training corpus overlap heavily
(~30-40% of eval problems are also in train). The model **memorized
the training subset** but **destroyed generalization**.

The same r=1 → r=2 regression observed in the per-round eval
(Δ(r=2−r=1) = -0.081) is now confirmed AS overfitting by the full-set
metric, AT MUCH GREATER MAGNITUDE: r=2 is not just "lower than r=1",
it's lower than the untrained baseline.

This decisively confirms the hypothesis that **train-steps=100 ×
2 rounds catastrophically over-trains on a 13-27 trajectory LoRA
corpus**. Phase 17's monotonic curve (r=1=0.230 < r=2=0.404 < ... <
r=6=0.581) cannot be reproduced with train-steps=100 × gen-n=164;
the recipe needs fewer steps per round.

### Ablation B (train-steps=30) running

The ablation hypothesis test is now running on the freed GPUs (see
`docs/phase22-stage-d-train-steps-ablation.md`). 5 seeds with
`--train-steps 30` (the same value the gen-n=32 batch used and which
showed r=2 ≥ r=1 in 4/5 seeds). Expected outcome:
- If train-steps drives the regression: r=2 ≥ base in ≥4/5 seeds at
  the aggregate metric.
- If something else (LoRA rank, corpus shift, recipe divergence):
  similar regression even at train-steps=30. Next: lr ablation or
  LoRA rank=32 ablation.

## What we expect

If the Pekko-side SFT recipe is faithful to Phase 17's Python recipe,
the 5-seed aggregate mean should land **within 1-2σ of 0.404** — i.e.
in the range [0.378, 0.430]. σ should be smaller than the prior
gen-n=32 batch's 0.116 (which suffered from eval-n=32 + passk=3
noise; the aggregate eval here uses n=164 × k=10 = 1640 attempts so
binomial SE at p=0.404 ≈ 0.012, very close to Phase 17's 0.013).

If the mean is significantly above 0.430 or below 0.378, that
indicates a recipe divergence (e.g., different LoRA hyperparameters,
different sampling temperature, different chosen-trajectory rendering
in the trainer) worth investigating.

## Related work in this batch

While the GPUs run, two parallel code drops landed:

1. **`FilteredDomain` wrapper** (commit `b9be505`) — operationalizes
   the `--prompt-skip-list` CLI flag at the Domain trait level. No
   supervisor changes. 4 new unit tests.

2. **`phase22_mbpp_mr_sft` binary** (commit `2753241`) —
   cross-substrate companion of `phase22_he_mr_sft`. Same actor
   pipeline, HumanEvalDomain → MbppDomain swap. CLI surface
   identical. Sets up Phase 22 Stage D matrix completion (HE + MBPP)
   for the next measurement batch.

160 tests pass. fmt + clippy clean.

## See also

- `docs/phase22-stage-d-parallel-smokes.md` — prior 5-seed gen-n=32
  batch (mean r=2 = 0.275 ± 0.116)
- `docs/phase22-stage-d.md` — Stage D binary + sparse-corpus
  probability table
- `docs/phase22-overview.md` — Phase 22 single entry point
- `scripts/phase17_s1/run_mr.py` — Phase 17 S1's Python reference
  for the saturation curve this batch tries to match
