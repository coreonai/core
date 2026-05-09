# Phase 13 S2 — Stage B 1M → 10M scale jump (honest negative: A1 confound)

Phase 13 design doc Stage B asked: does jumping from K9's tiny ~1M
char-model to a larger (~10M) model reduce the seed-to-seed
variance σ that Phase 13 S1 found at 1M?

S2 ran 5 seeds × {tiny (n_layer=4, n_embd=128, ~1M), medium
(n_layer=8, n_embd=256, ~10M)} with AdamW. **Result: an honest
negative driven by an unintended interaction with Phase 13 S1's
A1.**

## Setup

- Same K9 4-round protocol as Phase 13 S1 (1500 pretrain + 4
  rounds × 400 train_steps, gen_n=eval_n=24)
- New CLI flags `--n-layer / --n-head / --n-embd / --n-kv-head`
  let us scale the in-house GPT
- 5 seeds × {tiny (defaults: 4/4/128/4), medium (8/8/256/4)}
- 10 runs total, 2 GPUs parallel, ~25 min wallclock
- **Binary built AFTER A1 (challenge expansion 3 → 10)** —
  this is the confound

## Result

| metric | tiny (1M) | medium (10M) |
|---|---:|---:|
| mean_gen_pass | 0.013 ± 0.023 | 0.010 ± 0.010 |
| final_eval/24 | **0 ± 0** | **0.2 ± 0.45** |
| best_eval/24 | **1.8 ± 0.4** | **3.6 ± 2.9** |

Per-seed final_eval:
- tiny: [0, 0, 0, 0, 0]
- medium: [1, 0, 0, 0, 0]

Per-seed best_eval:
- tiny: [2, 2, 2, 2, 1]
- medium: [1, 5, 0, 6, 6]

## Confound — the A1 task expansion

Phase 13 S1's AdamW result (3-challenge K9): final_eval = 5.6 ± 3.4,
best_eval ≈ 9.8 ± 1.6.

S2 tiny (same model, **but 10-challenge K9**): final_eval = 0,
best_eval = 1.8 — *much weaker* than S1.

The difference is purely the challenge expansion (A1: 3 → 10):
- 3 challenges → 1M model can pretrain enough patterns to score 9-11/24
- 10 challenges → 1M model is undercapacity → fails almost completely

So **S2 isn't actually testing scale alone** — it's testing
"does 10× model capacity recover from 3× task expansion?". The
answer is "barely":
- 1M model on 10 challenges: best ≈ 1.8/24 (8%)
- 10M model on 10 challenges: best ≈ 3.6/24 (15%)
- 1M model on 3 challenges (Phase 13 S1): best ≈ 9.8/24 (41%)

10M doesn't catch up to 1M-on-3-challenges. Need 50M+ probably.

## Honest verdicts

**Variance reduction at 10M?** Inconclusive. σ(best_eval)
*increased* (0.4 → 2.9). σ(mean_gen) decreased (0.023 → 0.010).
σ(final_eval) was 0 at 1M (uniform failure) — meaningless.

**Mean improvement at 10M?** Yes for best_eval (+1.8), tiny for
final_eval (+0.2), no change for mean_gen.

**Stage B hypothesis ("σ shrinks with scale")?** Not directly
testable at this configuration because A1 confound dominates.

## What this teaches

1. **A1 challenge expansion was over-aggressive at 1M scale.** The
   pretrain budget (1500 steps × batch 64) is insufficient to
   memorize 10 distinct prompt patterns at this model size. Future
   measurements should match challenge count to model capacity.

2. **Two-axis interaction** (model size × task complexity) is
   significant. Naive single-axis scale-up to test variance fails
   when task complexity also moves.

3. **Phase 13 S1's 5.6 ± 3.4 final_eval was correctly variance-
   bounded** at the (1M, 3-challenge) configuration. That result
   stands.

## Phase 13 S3 plan — re-isolate scale

To answer "does scale-up reduce K9 σ?" cleanly:

**S3 option (a) — match challenges to S1**: rerun 5-seed × {tiny,
medium} with `--challenge-mask 0,1,2` (3 challenges, matching
S1). Variance comparison at fixed task complexity.

**S3 option (b) — increase pretrain budget for 10-challenge**:
medium 10M with pretrain_steps 1500 → 5000 + 600 → 1500
pretrain_examples. See if 10M can learn the 10-challenge surface.

**S3 option (c) — bigger model 50M+ on 10 challenges**: K8-style
nano_50m (already exists in `config.rs`) for K9. ~5× larger than
medium. Tests if our toy 1M-K9 scale was just too small for the
expanded task.

Recommended: **(a) first** (cheapest, isolates scale), then (b)
or (c) based on what (a) shows.

## Phase 13 design doc decision gate

Original gate:
> Stage B in Phase 13 design: B4 → measure Phase 11/12 results
> at 200M, see if they reproduce.

Revised gate after S2 outcome:
> Stage B at 200M is *premature* until we resolve the
> task-complexity × scale interaction. **Phase 13 S3 first
> isolates scale at the same task complexity** (option a above).
> Only after that do we know whether 200M is worth running.

## Risk #15 candidate

Adding to `docs/phase7-design.md`:

> **Task complexity × model scale interact non-trivially.**
> Phase 13 A1 (K9 challenge expansion 3 → 10) made the task
> too hard for 1M model and barely tractable for 10M; isolating
> "scale effect" requires fixed task complexity. When scaling
> any axis, sweep one at a time — interactions are not
> additive at toy scale.

## Reproducing

```bash
bash scripts/phase13_s2/run_tiny.sh    # GPU 0, 5 seeds × ~5 min
bash scripts/phase13_s2/run_medium.sh   # GPU 1, 5 seeds × ~5 min
python3 scripts/phase13_s2/analyze.py
```

## See also

- `docs/phase13-design.md` — Stage B was supposed to test variance
- `docs/phase13-s1-variance.md` — A2's clean variance bound at
  3-challenge / 1M; A1's challenge expansion broke that baseline
- `nanogpt-rs/src/config.rs::nano_50m` / `nano_300m` — existing
  scaled presets (Llama recipe at 50M)
