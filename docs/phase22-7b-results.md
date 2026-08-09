---
title: "Phase 22 — Qwen2.5-Coder-7B: migration + saturation + pass@k"
date: "2026-07-25"
---

# ⚠ Measurement corrections — read first

Two scoring defects, both found in late July 2026, invalidated a series of
conclusions in this document and its follow-ups. Each is fixed and
re-measured; this index exists so a reader does not have to reconstruct which
statement still stands. Full accounts in
`docs/phase22-c4-c5-rl-vs-sft.md` and `docs/phase22-c3-rl-step-semantics.md`.

**The defects.** (a) The RL loop verified *raw* completions while every other
consumer calls `domain.truncate_completion` — ~3× stricter at pass@1, and it
penalises long completions specifically. (b) `FilteredDomain` never delegated
`truncate_completion`, so wrapping a code domain in it turned truncation off
entirely — and every hard-tail experiment ran through `--prompt-skip-list`.
Same base, same problems, same sampling: pass@5 **0.4219 truncated vs 0.1562
un-truncated**.

| Conclusion as originally published | Status |
|---|---|
| "RL collapses on adapter sync" | **Retracted** — the defect was 256 optimizer updates per RL step; sync only made the drift visible |
| "The RL collapse is caused by an unbounded objective" | **Retracted** — no runaway exists; 8/8 runs rise once the reward is scored correctly |
| "RL is the weak axis / half of SFT" | **Retracted** — bounded RL matches SFT (+0.146 pass@5 / +0.227 pass@1 vs +0.145 / +0.203) |
| "Hard-tail SFT +0.254" | **Corrected to +0.145** pass@5 (+0.203 pass@1) on a consistent ruler |
| "Harvest is an inverted-U peaking at samples≈16" | **Retracted** — 6→16 helps, past 16 the mean is flat and only the variance grows |

**What survived unchanged**: the pass@k inference-time result (+0.347), the
full-set pass@1 SFT lift (+0.106), HumanEval+ (+0.088), and the 256-update
step-count defect and its fix.

**The transferable lesson**: every one of these came from comparing numbers
produced by two different scoring paths. Before comparing to a prior result,
re-measure the prior result on your ruler — `llm_actors::eval_sanity` and
`--sanity-strict` now enforce this in CI.

# TL;DR

Migrated the Rust/Pekko self-improve stack from Qwen2.5-Coder-**0.5B** to
**7B** (single A100 40GB), then measured what actually helps a strong 7B
base on HumanEval:

- **Multi-round SFT on full HumanEval: flat at pass@5 (saturated metric),
  but +0.106 at pass@1.** The per-round **pass@5** view is flat (baseline
  +0.037 ± 0.085, gentler −0.031) — but that metric is saturated (~0.64).
  Re-evaluating the *same* r2 checkpoints at **pass@1** (base 0.381, real
  headroom): **0.381 → 0.487, net +0.106 (4/4 seeds, ~2.3σ)**. **So SFT
  is NOT saturated on 7B — the "flat" was a pass@5 artifact.** (⚠ This
  supersedes the earlier "SFT ~0/saturated" reading below.)
- **pass@k (inference-time scaling): +0.347.** full-164 aggregate
  **pass@1 = 0.366 → per-prompt pass@10 = 0.713** (≈1.95×).
- **Self-SFT DOES work where there's headroom + harvest — the hard tail.**
  On HumanEval idx 100–163 (base pass@5 ~0.25, real headroom): samples=6
  gives a noisy +0.094 ± 0.106 (3/4 seeds); **samples=16 gives +0.254 ±
  0.068 (pass@5 0.246 → 0.500, 4/4 seeds, ~3.7σ)** — the first robust 7B
  self-improve win. ⚠ **Magnitude corrected to +0.145** (pass@5 0.422 →
  0.566) / **+0.203** (pass@1) on a consistent eval ruler — the base here
  was mis-measured. Still the robust win; see
  `docs/phase22-c4-c5-rl-vs-sft.md`.
- **Harvest: thin is worst, past ~16 the mean is flat and the variance
  grows.** ⚠ The original "interior optimum / inverted-U (6 → +0.094,
  16 → +0.254, 32 → +0.094)" is **retracted** — it was measured with
  truncation disabled. Re-scored on a consistent ruler: pass@5 0.500 / 0.566 /
  0.535 and pass@1 0.325 / 0.364 / **0.385** for samples 6 / 16 / 32. 6 → 16
  helps (paired +0.066 pass@5, 4/4 seeds); 16 → 32 is indistinguishable and
  the ranking **flips with the metric**, with 32 3–5× noisier. `samples≈16`
  stays the default for tightness, not for a higher mean.
- **RL (REINFORCE) on the hard tail COLLAPSES on adapter sync.** RLOO
  verifier-as-reward, k=4, sync-every=4: steps 0–3 healthy (~15/256 pass),
  then the first adapter sync craters it — lr=2e-4 → all seeds 0/256 (full
  collapse); lr=5e-5 → mean 15→3/256 (−80%, 1/4 still full-collapse).
  Gentler lr softens but doesn't prevent. RL is the *weak* axis; SFT
  (+0.254) is the robust hard-tail win. Reproduces Phase 22 Stage E.
  ⚠ **RETRACTED — sync is not the trigger.** See
  `docs/phase22-c3-rl-step-semantics.md`: the PG step was issuing **256
  optimizer updates per RL step** (a `--pg-micro-batch-size` memory knob
  leaking into the training math), so ~1024 updates had already ruined the
  policy before the first sync made it *visible* — steps 0–3 looked healthy
  only because the sampler was still on frozen base weights. Re-run with
  `--sync-every 1`, 2/2 seeds still collapse to 0/256 (at 1024 / 1280
  cumulative updates). C3 attributed the residual runaway to the unbounded
  negative-advantage CE ascent in `pg_sample_loss` — **that attribution is
  also retracted**, see the C4/C5 line below: the collapse was a scoring
  artifact, not an objective pathology.
  **C4/C5 follow-up** (`docs/phase22-c4-c5-rl-vs-sft.md`): the RL loop
  verified completions **without** `truncate_completion` while every other
  consumer applies it — a 3× reward-signal gap. With that fixed, **RL matches
  SFT**: positive-advantage-only gives **+0.152 pass@5 / +0.218 pass@1**
  (4 seeds) vs SFT's +0.145 / +0.203, though RL's σ is 3–4× wider (0.068 vs
  0.020), so SFT remains the better deployment choice. **The "RL collapses /
  runs away" narrative is retracted** — 8/8 runs rise once the reward is
  scored correctly; the earlier collapse was a length penalty in disguise
  (longer completions are likelier to emit a trailing top-level statement,
  which an un-truncated scorer counts as wrong).
- **Harder external benchmark (HumanEval+) — SFT WINS at the headroom
  metric.** EvalPlus's stricter tests (base pass@1 0.31 vs 0.37) give real
  full-benchmark headroom. MR-SFT (full 164, samples=6): at **pass@1**
  (where the headroom is) **0.326 → 0.413, +0.088 ± 0.0085 (4/4 seeds,
  ~10σ)**; at the saturated pass@5 metric only +0.051 (noisy). **Meta-lesson:
  the eval metric must match where the headroom is** — pass@5 masked the
  effect the whole way through the 7B study.
- **Verdict:** the MR-SFT recipe's worth is a function of **headroom ×
  harvest**, not the technique. No headroom (full set @ pass@5) → flat;
  headroom + adequate harvest (hard tail, samples ≥ 16) → a real lift
  (+0.145 pass@5 / +0.192 pass@1); thin harvest (samples=6) → weaker and
  noisier. Past ~16 more harvest neither helps nor hurts the mean but widens
  the spread — so pick 16 for tightness, not because it is a peak. (The
  earlier "tuned knob with a sweet spot / inverted-U" framing was a scoring
  artifact; see the harvest section.)
  Orthogonally, inference-time **pass@k** is always a training-free win
  (+0.347).

# Migration (code, all on origin/master)

| Piece | What | Commit |
|-------|------|--------|
| C1 | sharded-safetensors loader (`resolve_safetensors`, dual-mode file/dir) | `cbbfeb5` |
| C2 | `--model-dir` / `--model-id` override on the Phase-22 examples | `d954b98` |
| C3 | BF16 training path (`--train-bf16` / `--bf16`) | `fe63afd` |
| candle patch | vendored candle-core 0.10.2: matmul backward skips grads for frozen operands | `41ed3ce` |
| pipeline | CPU-side LoRA merge + free-before-load reload | `0ca067f` |
| trainer split | `--trainer-gpu` (inference + trainer on separate cards) | `d0b058c` |

Config/dimension adaptation is automatic (config.json → Qwen2Config;
tied/untied lm_head runtime-detected — 7B is untied, 0.5B tied).

Upstream: the candle fix is submitted as **huggingface/candle#3773**; drop
the `[patch.crates-io]` + `vendor/candle-core-0.10.2/` once it merges.

# Memory findings (measured on 7B, bf16)

- **Inference:** 15.1 GB (fits one 40GB card).
- **LoRA training multiplier:** candle 0.10.2 matmul backward computes+stores
  a gradient for BOTH operands unconditionally → for LoRA (~all weights
  frozen) it allocated a full weight-sized grad per frozen matmul → **~4×
  base** peak (0.5B 4.36GB, 1.5B 12.4GB; 7B predicted ~60GB → OOM). Patch
  (`if operand.track_op()`) cuts it to ~1.2× → **7B training 15.2GB**.
  Loss bit-identical; 179 tests pass.
- **Full pipeline residency:** inference model (15GB) + trainer (15GB) =
  ~30GB co-resident. Fixes: LoRA merge moved to CPU (was a 3rd 15GB GPU
  copy); `handle_reload` frees the old model before loading (was 3×15=45GB).
- **Training batch × seq:** batch=4 SFT on real HumanEval seqs (152K-vocab
  all-position logits + 28-layer MLP activations) peaks ~39GB **even with
  the trainer alone** on a card. Working config = **`--trainer-gpu` (2 GPUs)
  + `batch_size=2`** → trainer peak 27.8GB, model 15.1GB.

# Experiment: multi-round SFT saturation (4 seeds, rounds=3)

Recipe: Phase-17 (samples-per-prompt=6 → 984 attempts/round, max-new=200,
fresh-opt, non-cumulative harvest, top_k=0), two-GPU + batch=2, per-round
eval n=40 × passk=5 (directional; wide σ). base 0.638 ± 0.083 (same model).

**Baseline** (train-steps=100, lr=2e-4):

| Seed | base | r0 | r1 | r2 |
|------|------|----|----|----|
| 42 | 0.750 | 0.750 | 0.675 | 0.750 |
| 100 | 0.575 | 0.650 | 0.650 | 0.625 |
| 200 | 0.575 | 0.625 | 0.625 | 0.725 |
| 300 | 0.650 | 0.600 | 0.675 | 0.600 |
| **mean±σ** | 0.638±.083 | 0.656±.066 | 0.656±.024 | **0.675±.074** |

net Δ(base→r2) = **+0.037 ± 0.085** (< 0.5σ, not significant).

**Gentler** (train-steps=30, lr=5e-5):

| Seed | base | r0 | r1 | r2 |
|------|------|----|----|----|
| 42 | 0.750 | 0.725 | 0.725 | 0.700 |
| 100 | 0.575 | 0.550 | 0.550 | 0.550 |
| 200 | 0.575 | 0.550 | 0.575 | 0.600 |
| 300 | 0.650 | 0.675 | 0.550 | 0.575 |
| **mean±σ** | 0.638±.083 | 0.625±.089 | 0.600±.084 | **0.606±.066** |

net Δ(base→r2) = **−0.031 ± 0.043**. Gentler does *worse* → **over-training
ruled out; the base is saturated.**

# Experiment: pass@k (inference-time scaling)

`phase22_humaneval_baseline --model-id Qwen2.5-Coder-7B --n-problems 164
--passk 10 --sequential --aggregate` (8-GPU split by `--offset`):

| Metric | 7B | 0.5B (Phase 17 S6) |
|--------|----|--------------------|
| aggregate pass@1 | **0.366** (600/1640) | 0.216 |
| per-prompt pass@10 | **0.713** (117/164) | 0.524 |
| lift | **+0.347 (≈1.95×)** | +0.308 |

Headroom is in the hard tail: per-prompt pass@10 by problem index runs
~1.0/0.95 for idx 0–40 down to 0.43–0.59 for idx 100–164 — where any future
training gains would have to come from.

# Conclusions

1. The MR-SFT recipe that compounds a weak base (0.5B: 0.23→0.48) does
   **nothing** on a strong 7B base (HumanEval-saturated) — at either
   learning strength. This is **saturation, not over-training**.
2. **pass@k is the win on 7B** (+0.347), training-free — value comes from
   inference-time scaling.
3. Practical 7B recipe on 40GB: bf16, two-GPU (`--trainer-gpu`), batch=2,
   patched candle-core (or wait for #3773).

# Experiment: hard tail (headroom) + cold-start rescue

HumanEval idx 100–163 (64 hard problems, base pass@5 ~0.25 — real headroom),
via `--prompt-skip-list 0..99` (FilteredDomain). Same two-GPU + batch=2
recipe, rounds=3, 4 seeds, per-round eval on all 64 at passk=5.

**samples-per-prompt = 6** (thin harvest, ~18–27 pairs/round):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.203 | 0.188 | 0.406 | +0.172 |
| 100 | 0.281 | 0.172 | 0.359 | 0.406 | +0.125 |
| 200 | 0.234 | 0.266 | 0.234 | 0.172 | −0.062 |
| 300 | 0.234 | 0.312 | 0.266 | 0.375 | +0.141 |
| **mean** | 0.246 | 0.238 | 0.262 | **0.340** | **+0.094 ± 0.106** |

Real but noisy lift (3/4 up, one declines) — headroom present, but the base
rarely passes the hard problems → sparse harvest → high variance
(cold-start, Phase 9 S5 risk #11).

**samples-per-prompt = 16** (rich harvest — base pass@1 ~0.15 → P(≥1 of 16
pass) ≈ 0.93 per problem):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.188 | 0.516 | 0.516 | +0.282 |
| 100 | 0.281 | 0.359 | 0.500 | 0.547 | +0.266 |
| 200 | 0.234 | 0.234 | 0.234 | 0.391 | +0.157 |
| 300 | 0.234 | 0.312 | 0.344 | 0.547 | +0.313 |
| **mean** | 0.246 | 0.273 | 0.399 | **0.500** | **+0.254 ± 0.068** |

⚠ **The +0.254 figure is inflated — on one consistent eval ruler it is
+0.145.** See `docs/phase22-c4-c5-rl-vs-sft.md`: re-scoring these *same*
saved r2 checkpoints through the `EvalSequential` path gives 0.566, but the
**base** measures 0.422 there, not 0.246. The r2 endpoint reproduces; the
base does not, so the gain was inflated ~1.75× by pairing a mis-measured
base with a sound r2. Corrected: **pass@5 0.422 → 0.566 = +0.145** (4 seeds,
σ 0.020), **pass@1 0.172 → 0.364 = +0.203**. Base re-verified on a second
independent draw (passk=10, 640 samples: pass@1 0.161). Direction and
robustness of the win are unchanged; only the magnitude is. The mechanism
behind the 0.246 is unrecovered — these runs were launched ad hoc and no
command line survives.

**Cold-start rescue works decisively:** mean lift more than doubled
(+0.094 → +0.254; hard-tail pass@5 **doubled** 0.246 → 0.500), σ tightened
(0.106 → 0.068), **4/4 seeds positive** (incl. the samples=6 dissenter,
which sat at base through r0/r1 then jumped +0.156 once its harvest grew).
Harvest self-reinforces (18 → 200 for the strongest seed). ~3.7σ significant
— **the first robust self-improve win at 7B.**

**samples-per-prompt = 32** (2× the harvest again — is more better?):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.125 | 0.328 | 0.375 | +0.141 |
| 100 | 0.281 | 0.266 | 0.391 | 0.344 | +0.063 |
| 200 | 0.234 | 0.234 | 0.328 | 0.453 | +0.219 |
| 300 | 0.234 | 0.422 | 0.422 | 0.188 | −0.046 |
| **mean** | 0.246 | 0.262 | 0.367 | **0.340** | **+0.094 ± 0.113** |

**⚠ RETRACTED — the inverted-U was a scoring artifact.** The table below was
measured with `FilteredDomain` silently disabling completion truncation (see
`docs/phase22-c4-c5-rl-vs-sft.md`). All three r2 checkpoint sets have now been
re-scored on the unfiltered path; the symmetry that made it look like a clean
inverted-U does not survive.

*Old (un-truncated) ruler:*

| samples/prompt | r2 mean | net Δ (base→r2) |
|----------------|---------|-----------------|
| 6 | 0.340 | +0.094 ± 0.106 |
| **16** | **0.500** | **+0.254 ± 0.068** |
| 32 | 0.340 | +0.094 ± 0.113 |

*Re-measured on the consistent ruler (same r2 checkpoints, 4 seeds, 64
hard-tail problems, base 0.4219 pass@5 / 0.1719 pass@1):*

| samples/prompt | pass@5 | Δ | pass@1 | Δ |
|----------------|--------|---|--------|---|
| 6 | 0.500 ± 0.038 | +0.078 | 0.325 ± 0.062 | +0.153 |
| **16** | **0.566 ± 0.020** | **+0.145** | 0.364 ± 0.037 | +0.192 |
| 32 | 0.535 ± 0.090 | +0.113 | **0.385 ± 0.108** | **+0.213** |

What actually holds:

- **6 → 16 helps** at pass@5: paired +0.066, 4/4 seeds positive (t = 2.50,
  df = 3). At pass@1 the same comparison is +0.039 (t = 0.92) — not resolved.
- **16 → 32 is indistinguishable, and the ranking flips with the metric.**
  pass@5 favours 16 by +0.031 (t = 0.59); pass@1 favours **32** by 0.021
  (t = −0.32). Neither is significant at n = 4.
- **32 is much noisier**: σ 0.090 / 0.108 versus 16's 0.020 / 0.037, a 3–5×
  spread. So "too much harvest destabilises" survives *as a variance claim*
  — but not as a mean-degradation claim.
- The old "samples=32 lands exactly back at samples=6" symmetry is gone: on
  the consistent ruler 32 beats 6 on both metrics.
- The specific collapse cited as the mechanism — "seed 300 collapsed
  0.422 → 0.188" — inverts: **that seed is the best checkpoint of any harvest
  setting** on the consistent ruler (pass@5 0.641). It produced longer
  completions, which an un-truncated scorer punishes as wrong. Same mechanism
  that made the RL `fulladv` arm look like it was collapsing.

### Round-by-round, re-measured (the collapse never existed)

Every r0/r1 checkpoint was re-scored too, so the whole curve sits on one
ruler rather than just its endpoint (mean pass@5 ± σ over 4 seeds):

| samples | r0 | r1 | r2 |
|---------|----|----|----|
| 6 | 0.434 ± 0.047 | 0.477 ± 0.079 | 0.500 ± 0.038 |
| 32 | 0.449 ± 0.045 | 0.531 ± 0.065 | 0.535 ± 0.090 |

(base = 0.4219; samples=16 was re-scored at r2 only, 0.566 ± 0.020.)

- **Both settings rise monotonically**, and per-seed net r0→r2 is positive in
  7 of 8 runs (6: +0.031 / +0.156 / 0 / +0.078; 32: +0.063 / +0.125 / +0.016
  / +0.141). There is no round at which samples=32 turns over.
- The old ruler reported samples=32 going *backwards* at r2 (r1 0.367 → r2
  0.340) with "seed 300 collapsed 0.422 → 0.188". On the consistent ruler
  that seed reads **0.500 → 0.609 → 0.641** — monotone up, and the best
  checkpoint in the entire harvest sweep.
- r0 sits essentially at base (0.434 / 0.449 vs 0.4219), which is the
  expected shape after a single round and a useful sanity check on the
  re-scoring.

So "too much harvest over-trains and degrades" had **no round-level support
either** — the mean is flat past 16 and only the variance grows.

**Corrected reading**: harvest matters, thinly-harvested (6) is worst, and
past ~16 the mean is flat while the variance grows. "Interior optimum /
inverted-U" overstates what the data supports; `samples≈16` remains a
reasonable default, chosen for **tightness** rather than a higher mean.

# Experiment: RL (REINFORCE) on the hard tail

Ported `phase22_he_reinforce` to 7B + hard tail (`--model-id`, `--train-bf16`,
`--trainer-gpu`, `--prompt-offset`; commit `160ccd6`). RLOO verifier-as-reward,
idx 100–163, k=4, max_new=192, pg-micro-batch=1, sync-every=4, 4 seeds.
Validated: peak model GPU 15.1GB / PG-trainer GPU 20.2GB (fits 40GB).

Per-step passes / 256 (all 4 seeds sample the hard tail each step):

| step | 0 | 1 | 2 | 3 (sync fires) | **4 (first post-sync)** |
|------|---|---|---|---|---|
| lr=2e-4 (seed 42) | 17 | 18 | 17 | 13 | **0** → 0 for all remaining 16 steps |
| lr=5e-5 mean (4 seeds) | 16 | 18 | 15 | **15** | **3** (seed 100 → 0, others 2/3/7) |

**The first adapter sync craters the policy.** Steps 0–3 have a healthy,
non-zero gradient signal (~15/256 ≈ 6% per-completion pass, plausible for the
hard tail). At the first sync (`SaveMergedCheckpoint` → `ReloadCheckpoint`,
step 3) the sampling model reloads the merged LoRA and the policy collapses:
lr=2e-4 → **0/256 for every remaining step** (full mode collapse); lr=5e-5 →
mean drops 15→3 (−80%), one seed to 0, the rest barely alive. Gentler lr only
softens it. **RL + adapter-sync is unstable at this scale/sparsity** — the same
mode-collapse Phase 22 Stage E found on the full set. Infra works; the RL
*algorithm* is the weak axis. **SFT's +0.254 is the robust hard-tail win.**

# Experiment: harder external benchmark (HumanEval+)

EvalPlus's HumanEval+ is a **zero-code drop-in** — same jsonl schema
(`{task_id, prompt, entry_point, test}`, `test` defines `check(candidate)`),
so `HumanEvalDomain` parses it via `--jsonl` (needs numpy for the tests).
Saved to `data/humanevalplus/HumanEvalPlus.jsonl` (164 problems, ~77KB
tests each). Genuinely harder — 7B base:

| metric | HumanEval+ | standard | Δ |
|--------|-----------|----------|---|
| aggregate pass@1 | 0.307 | 0.366 | −0.059 |
| per-prompt pass@10 | 0.634 | 0.713 | −0.079 |

MR-SFT (full 164, rounds=3, samples=6, two-GPU + bf16, 4 seeds):

- **pass@5** (per-round training metric, base ~0.59 — saturated): net
  **+0.051 ± 0.047** (3/4 seeds up). Modest, cleaner than standard
  HumanEval's +0.037, but the metric masks the headroom.
- **pass@1** (full-164 aggregate on the r2 checkpoints — the headroom
  metric, base 0.326):

  | | pass@1 |
  |---|---|
  | base | 0.326 |
  | r2 seeds | 0.424 / 0.413 / 0.404 / 0.412 |
  | **r2 mean** | **0.413 ± 0.0085** |
  | **net Δ** | **+0.088 ± 0.0085 (4/4 seeds, ~10σ)** |

**SFT clearly helps on a genuinely harder external benchmark** — +0.088
at pass@1, all four seeds tightly clustered. The pass@5 view understated
it 2× because that metric is saturated (~0.64). **Meta-lesson: match the
eval metric to where the headroom is** — pass@5 masked the SFT effect
throughout the 7B study (the saturated full-HumanEval "flat" result is
partly a pass@5 artifact; pass@1 has headroom the whole time).

# Where next (not done)

- ~~Re-eval standard-HumanEval SFT at pass@1~~ **DONE**: base 0.381 → r2
  mean 0.487, **+0.106 (4/4 seeds)** — confirmed the "flat" was a pass@5
  artifact. SFT helps 7B on HumanEval across the board at pass@1 (full
  +0.106, HumanEval+ +0.088, hard tail +0.254).
- ~~**Fix RL collapse**: reference-policy KL, off-policy correction,
  no-sync + one final merge, or SFT-warmstart.~~ **DIAGNOSED** in
  `docs/phase22-c3-rl-step-semantics.md` — the collapse is an
  optimizer-step-count bug (256 updates/RL step) on top of an unbounded
  REINFORCE objective, *not* adapter sync. Step-count fixed; the remaining
  work is bounding the objective (positive-advantage-only first, then KL /
  PPO clip) and re-testing the fixed path at equal update dose.
- Even harder benchmarks (LiveCodeBench / BigCodeBench) for more headroom.
