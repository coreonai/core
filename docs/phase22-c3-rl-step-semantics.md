---
title: "Phase 22 follow-up C3 — the 7B RL collapse is an optimizer-step-count bug, not adapter sync"
date: "2026-07-28"
---

# ⚠ Superseded in part (2026-07-30)

The step-count defect below is real and the fix stands. **But this document's
root-cause attribution — "an unbounded objective, amplified by the update
count" — is retracted.** `docs/phase22-c4-c5-rl-vs-sft.md` found that the RL
loop scored completions *without* `domain.truncate_completion` while the eval
applies it. That penalised long completions for being long rather than wrong,
which is exactly the "collapse toward / away from EOS" signature reported
here. With the reward scored the same way the eval scores it, **8/8 runs rise
and the full-advantage arm never collapses** — including at 30 updates, and
including the configuration this document calls fatal. Read the collapse
tables below as "what an un-truncated scorer reports", not as policy
behaviour.

# TL;DR

`docs/phase22-7b-results.md` concluded that REINFORCE on the 7B hard tail
"COLLAPSES on adapter sync" — healthy for steps 0–3, then 0/256 forever
once `SaveMergedCheckpoint → ReloadCheckpoint` fires. **That reading was
wrong**, and so was the first replacement hypothesis. Measured here:

- **The PG step was issuing 256 optimizer updates per RL step.**
  `--pg-micro-batch-size 1` was introduced as a *memory* knob, but
  `train_qwen_lora_pg_step` called `backward_step` once **per micro-batch**.
  At 64 prompts × k=4 that is 256 AdamW updates per RL step — ~1024 before
  the first `--sync-every 4` sync. An entire SFT round uses 30. AdamW
  normalises by gradient magnitude, so the numerically tiny PG loss
  (−0.03) is *not* a small step.
- **Adapter sync is not the trigger.** Re-run with `--sync-every 1` (sync
  after every step, so the sampler is never more than one step stale):
  **both seeds still collapse to 0/256**, at 1024 and 1280 cumulative
  updates. Frequent syncing only moves *when*. The original "collapse at
  the first sync" was the first time the sampler *saw* a policy that
  1024 updates had already ruined — before that it sampled frozen base
  weights, which is why steps 0–3 looked flat and healthy.
- **The root cause is an unbounded objective, amplified by the update
  count.** `pg_sample_loss` computes `mean_ce * reward`; a
  negative-advantage sample is therefore *gradient ascent on
  cross-entropy*, which has no upper bound. Under RLOO with k=4, ~75% of
  the surviving samples carry negative advantage. Nothing anchors the
  policy, so with enough updates it runs away — in either direction:

  | run | dose | pass | `comp_len` | signature |
  |-----|------|------|-----------|-----------|
  | original (`--sync-every 4`) | 1024 behind one sync | 0/256 | short, 267s/step | collapse *toward* EOS |
  | C3 legacy seed 100 (`--sync-every 1`) | 1024 cumulative | 0/256 | **192.0 = `max_new` ceiling**, 1200s/step | collapse *away from* EOS |
  | C3 legacy seed 42 (`--sync-every 1`) | 1280 cumulative | 0/256 | **192.0 = ceiling**, 1318s/step | collapse *away from* EOS |

- **Fix shipped** (one optimizer update per RL step + drop RLOO
  zero-advantage samples). Both fixed seeds survive 6 steps with no
  degradation — but that is **6 updates**, so this run demonstrates
  *survival, not improvement*, and does not prove the runaway is cured.
  It removes the 256× amplifier; it does not bound the objective.

# The defect

`llm-actors/src/qwen2_lora.rs`, before this change:

```rust
for chunk in samples.chunks(mb) {
    ...
    optimizer.backward_step(&loss)?;   // <-- one AdamW update PER CHUNK
}
```

Micro-batching exists to bound peak GPU memory when completions are long
(`max_new ≥ 64`). It should not change the *math* of a step. It did:

| setting | samples | chunks | AdamW updates per RL step |
|---------|---------|--------|---------------------------|
| 7B hard tail, `--pg-micro-batch-size 1` | 64 × 4 = 256 | 256 | **256** |
| an entire Phase-17 SFT round | — | — | 30 |

Two aggravating details:

1. **Zero-advantage samples still moved the weights.** Under RLOO a prompt
   whose k completions all share a verdict gets advantage exactly 0 — on
   the hard tail that is ~78% of the batch (measured: `used/skip = 56/200`).
   Their gradient is all zeros, but `AdamW::step` still applies
   `m_hat/(√v_hat + eps)` from the momentum tail, so each one nudged the
   weights with no signal behind it. They also each cost a full
   forward+backward.
2. **The damage was invisible until a sync.** `QwenModelActor` samples from
   the base weights; the LoRA delta lives in `QwenTrainerActor`. With
   `--sync-every 4`, steps 0–3 all sample the *same frozen base policy* —
   which is why they read flat and healthy while the trainer drifted 1024
   steps away.

# The fix

`train_qwen_lora_pg_step_cfg` + `PgStepConfig` / `PgStepStats`:

- **`accumulate_grads`** (default `true`) — sum the per-chunk gradients and
  issue a single `Optimizer::step` per RL step. Micro-batching is a pure
  memory knob again. Unit-tested: `micro_batch ∈ {0, 1, 2}` now land on
  **identical** weights.
- **`skip_zero_advantage`** (default `true`) — drop `|advantage| ≈ 0`
  samples before the forward pass. On the hard tail this also removes ~78%
  of the backward work.
- An all-zero-advantage step (no prompt had a mixed verdict) is a logged
  no-op, not an error that kills the run.
- New per-step diagnostics: `upd=`, `used/skip=`, `comp_len=`. **`comp_len`
  is the collapse tell-tale** — it hits either 0 or the `max_new` ceiling.

Only the LoRA `Var` gradients are accumulated between chunks. A candle
`GradStore` also holds a gradient for every intermediate activation, so
retaining whole stores across chunks OOMs a 7B backward at `max_new=192`
(hit on the first attempt); the last chunk's store is reused as the carrier
handed to `Optimizer::step`, keeping peak memory at one chunk's backward.

Old semantics stay reachable via `--pg-legacy-updates` /
`--pg-keep-zero-advantage` so the collapse remains reproducible.

# Experiment: A/B on the 7B hard tail

`scripts/phase22_c3/rl_step_semantics_ab.sh` — 2 arms × 2 seeds, 7B, idx
100–163, k=4, `max_new=192`, `--pg-micro-batch-size 1`, lr=2e-4,
**`--sync-every 1`** (so any damage shows at the very next step), 6 steps,
4 runs × 2 GPUs.

`cum_upd` = optimizer updates applied *before* that step's generation, i.e.
the dose the sampled policy actually carries.

| step | fixed s42 | fixed s100 | legacy s42 | legacy s100 |
|------|-----------|------------|------------|-------------|
| 0 | 17 (0 upd) | 16 (0) | 13 (0) | 16 (0) |
| 1 | 21 (1) | 19 (1) | 19 (256) | 3 (256) |
| 2 | 13 (2) | 24 (2) | 28 (512) | 3 (512) |
| 3 | 9 (3) | 19 (3) | 13 (768) | 13 (768) |
| 4 | 7 (4) | 13 (4) | 20 (1024) | **0 (1024)** |
| 5 | 14 (5) | 23 (5) | **0 (1280)** | **0 (1280)** |

(pass count out of 256.)

## The noise floor — read this before believing any single number

The original runs hand it over for free: with `--sync-every 4`, steps 0–3
all sample the **same frozen base policy**, so those 16 readings (4 seeds ×
4 steps) measure one policy repeatedly:

> **base = 16.4/256, σ = 4.5, range 9–28, 2σ band 7.3–25.4**

(The lr=5e-5 rerun's steps 0–3 are byte-identical — sampling from base with
the same seeds — so it is not double-counted.)

Consequences:

- **Every fixed-arm reading (7–24) is inside the band.** The fixed arm shows
  no degradation and no improvement. With 6 updates total, that is exactly
  what it was designed to test — survival.
- Legacy seed 42's peak of 28 is marginally outside on 1 of 5 readings.
  Not a claim.
- `0` and `3` are far outside. Those are real.
- A prompt-level flake also exists: at step 0 no training has happened and
  `comp_len` is byte-identical across arms (generation is deterministic),
  yet seed 42 scored 13 (legacy) vs 17 (fixed) on the *same* completions —
  the 8s `python3` verify timeout (`domain/human_eval.rs:83`) tripping under
  8-GPU load. A few counts of every pass number are verifier noise.

# Conclusions

1. **Retract "RL collapses on adapter sync"** (`a44955c`,
   `docs/phase22-7b-results.md`). Sync is not the trigger: with syncs every
   single step, 2/2 seeds still collapse to 0/256. Sync cadence only
   controls *when* the already-drifted policy becomes visible, and how far
   it drifts between corrections.
2. **The proximate bug is real and measured**: 256 unintended optimizer
   updates per RL step, an artifact of a memory knob leaking into the
   training math. Fixed, with the equivalence property now under test.
3. **The root cause is not fixed.** REINFORCE with unbounded
   negative-advantage CE ascent has no anchor; the update-count fix removes
   a 256× amplifier but not the runaway itself. The fixed arm's 6 updates
   are ~200× fewer than the dose that kills the legacy arm, so it has not
   been tested at equal dose.
4. **CLAUDE.md gotcha #9 again** — a Pekko-driven mechanism diverged from
   its reference recipe, and the cause was in the inner training step, not
   the actor wiring. Found by reading `train_qwen_lora_pg_step` before
   spending a GPU-hour.

# Where next

- **Bound the objective** — the actual fix. Cheapest first: positive-
  advantage-only (drop `reward < 0`), which removes the unbounded term
  outright and reduces to rejection-sampling FT / RAFT — the same family as
  the SFT recipe that already gives **+0.254** on this hard tail. Then
  reference-policy KL, or PPO-style ratio clipping (needs old log-probs).
- **Test the fixed arm at equal dose** (30–50 steps ≈ 30–50 updates, ~8–12
  GPU-h for 4 runs) to find out whether one-update-per-step *learns*, or
  merely runs away 256× slower. Either answer is worth having; neither is
  established by the 6-step run.
- Whatever the RL outcome, SFT remains the robust hard-tail win (+0.254).
  RL is still the weak axis — but now for a stated, testable reason.
