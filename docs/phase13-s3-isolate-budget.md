# Phase 13 S3 — Isolate scale + budget fix (chain of honest negatives → K9 retired)

S3 ran two parallel measurements:

- **(a) S3-isolate**: 5 seeds × {tiny 1M, medium 10M} with
  `--challenge-mask 0,1,2` (3-challenge fixed, matching S1 task
  complexity). Tests Stage B's "scale shrinks variance" hypothesis
  cleanly.
- **(b) S3-budget**: 5 seeds × medium 10M with **5000 pretrain
  steps** (3.3× S2's 1500) on the full 10-challenge K9. Tests
  whether the A1 confound is curable with more compute.

Result: a chain of honest negatives that together motivate
**retiring K9 1M as the substrate for fine-grained algorithmic
comparisons**.

## Setup

- 4 GPUs in parallel: 0=isolate_tiny, 1=isolate_medium,
  2=budget seeds 0-2, 3=budget seeds 3-4
- ~30 min wallclock total
- All AdamW, K9 RustCode, 4 rounds × (24 gen / 24 eval / 400
  train_steps)

## Results

### (a) S3-isolate at 3-challenge K9

| metric | tiny (1M) | medium (10M) |
|---|---:|---:|
| mean_gen_pass | 0.027 ± 0.033 | 0.033 ± 0.047 |
| final_eval/24 | 1.2 ± 0.4 | 1.2 ± 0.8 |
| best_eval/24 | **3.0 ± 1.9** | **1.6 ± 0.5** |

Per-seed best_eval:
- tiny: [2, 1, 5, 2, 5]
- medium: [1, 1, 2, 2, 2]

### (b) S3-budget at 10-challenge K9 with 5K pretrain

| metric | budget_medium |
|---|---:|
| mean_gen_pass | **0.071 ± 0.034** |
| final_eval/24 | 0.4 ± 0.9 |
| best_eval/24 | **5.2 ± 3.1** |

Per-seed best_eval: [8, 2, 9, 4, 3]

## Three honest negatives in this measurement

### 1. Scale doesn't shrink variance at 3-challenge (S3a)

S3a was supposed to cleanly test the Stage B hypothesis "σ
shrinks at 4× model scale". Result:

- σ(best_eval): 1.9 → 0.5  (medium 0.29× tiny — looks like a win)
- mean(best_eval): 3.0 → 1.6  (medium *worse* — under-trained)

So "less variable but uniformly weaker" — 10M model with the same
1500-step pretrain budget under-fits relative to 1M. The reduced
variance is a side effect of converging to a uniformly-mediocre
solution rather than finding a good one some seeds.

This isn't "scale reduces noise"; it's "10M needs more compute to
reach 1M's competence". Can't isolate scale from compute budget.

### 2. S3a tiny baseline ≠ Phase 13 S1 baseline (cross-batch σ)

Same config, different fresh-seed pretrains:
- Phase 13 S1 (3-ch tiny, AdamW): best_eval **9.8 ± 1.6**
- Phase 13 S3a (3-ch tiny, AdamW): best_eval **3.0 ± 1.9**

Both 5 seeds. **The means differ by 6.8/24 (28%)** — far larger
than the within-batch σ of 1.6. Same code, same task, same model
config; just different fresh-seed paths and different timing.

Implication: K9 1M's variance is **not just within-seed-batch but
also cross-batch**. 5-seed measurement gives a tight σ within a
batch but the batch-level mean itself is noisy. The "σ ≈ 3.4 / 24"
estimate from Phase 13 S1 was an *underestimate*; the true
variance across runs is larger.

This makes risk #14 stronger: not even 5 seeds are enough at K9
1M. Single-batch claims need cross-batch replication too.

### 3. Budget partially closes A1 gap but doesn't reach S1 baseline

- S2 medium (10-ch, 1500 pretrain): best 3.6 ± 2.9
- **S3b medium (10-ch, 5000 pretrain): best 5.2 ± 3.1** (Δ +1.6)
- S1 baseline (3-ch tiny, 1500 pretrain): best 9.8 ± 1.6

Tripling pretrain budget at 10M scale recovers about half the gap
to S1's 3-challenge result. The 10-challenge task is genuinely
harder for a 10M-class model — even 3.3× compute doesn't fully
close it. To match the S1 (3-ch) baseline at the 10-challenge
task, we'd need either a bigger model or much more pretrain.

## Meta-finding — K9 1M is the wrong measurement substrate

Phase 13 S1 (variance bound) + S2 (scale jump) + S3 (isolate +
budget) together demonstrate:

1. **K9 21-prompt eval** has σ ≈ 3.4 / 24 within a 5-seed batch
   AND σ ≈ 6-7 / 24 across batches.
2. **K9 1M model** is at the bottom of its task-capacity range —
   small changes (10 vs 3 challenges, 1.5K vs 5K pretrain) push
   it from "barely working" to "fully failing".
3. **Scale jumps** (1M → 10M) at fixed compute budget *under-train*
   the bigger model and don't yield clean variance reduction.
4. **Cross-batch noise** is bigger than within-batch noise.

Net: K9 at 1M is **smoke-test infrastructure**, not measurement
substrate. Algorithmic comparisons (Muon vs AdamW, DPO variants,
OPD) at K9 1M produce mostly noise. The Phase 11–12 single-run
claims (now retracted as 1σ-noise candidates) and Phase 13's
own measurements collectively show that **claims at K9 1M scale
should be considered observational, not validated**.

## Phase 13 conclusion

**Stage A (variance bound)**: complete. A1 made K9 too hard for
1M; A2 quantified σ at 3-ch but A3-onwards is moot at this
substrate.

**Stage B (200M)**: deferred. The Stage B hypothesis ("σ shrinks")
is unanswerable cleanly at the 1M → 10M jump because of compute-
budget × scale interaction. At 200M the same problem amplifies.

**Stage C (Qwen 1B)**: **promoted to next priority.** Phase 9 S4
and S5 already demonstrated meaningful signals at Qwen 0.5B (sum-
AUC 0.702, +33pp self-improve). Real-world models + real-world
benchmarks (HumanEval) sidestep K9's noise floor entirely.

**Stage D (in-house 500M+)**: indefinitely deferred. Without a
clean Stage B win, the engineering investment (bf16, gradient
checkpointing, multi-GPU sharding) doesn't pay back.

## Risk register update

Risk #14 strengthened (cross-batch σ): K9 1M σ ≈ 3.4 within-batch
**and** ≈ 6-7 cross-batch. Single-batch 5-seed claims still need
cross-batch replication to be production-confident.

Risk #15 confirmed (compute-budget × scale interaction): scaling
the model up at fixed train budget under-trains the bigger model.
When scaling N×, also scale compute and seeds proportionally.

## What we learned from Phase 13

The deepest finding isn't any of the individual measurements but
the **substrate-level lesson**:

> Phase 11–12's single-run K9 claims (DPO collapse,
> hybrid α=0.3 r1=18/24, Muon +78% gen) and Phase 13's own
> measurements all sit at the same noise floor. K9 at 1M is not a
> reliable enough measurement vehicle to distinguish algorithmic
> deltas at this granularity. Future algorithmic comparisons
> should target either richer real-world benchmarks (HumanEval/
> MBPP via Phase 9 path) or bigger models where noise dampens.

This commit closes Phase 13 with that conclusion. **Project goes
to Stage C (Qwen + real benchmarks) for any further algorithmic
comparison.**

## Reproducing

```bash
bash scripts/phase13_s3/run_isolate_tiny.sh    # GPU 0, 5 seeds
bash scripts/phase13_s3/run_isolate_medium.sh   # GPU 1, 5 seeds
bash scripts/phase13_s3/run_budget_medium.sh 2 "0 1 2"  # GPU 2
bash scripts/phase13_s3/run_budget_medium.sh 3 "3 4"    # GPU 3
python3 scripts/phase13_s3/analyze.py
```

## See also

- `docs/phase13-design.md` — original 4-stage plan; Stage B/D
  deferred per this doc
- `docs/phase13-s1-variance.md` — first variance bound (within-batch)
- `docs/phase13-s2-scale.md` — scale jump confounded by A1
- `docs/phase7-design.md` — risk #14 strengthened, #15 confirmed
- Phase 9 S4/S5 (Notion) — Stage C precedent: Qwen + real benchmarks
