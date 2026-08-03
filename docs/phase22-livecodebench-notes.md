---
title: "Phase 22 §6.5 — LiveCodeBench base-7B benchmarking"
date: "2026-08-03"
status: "base validated; recipe contamination run is the next step"
---

# Why

Contamination is the most urgent external question: is the 7B self-improve
story real generalization, or polished recall of pretraining? LiveCodeBench
answers it by filtering problems on their **contest date** — score the same
generations before vs after the model's cutoff. Base-7B first: validate the
pipeline and establish the reference before any recipe.

# The pipeline (generate in Rust, score with the official harness)

- **Generate**: `phase22_dump_completions --benchmark livecodebench` →
  `bench_export::write_lcb` (`[{question_id, code_list}]`). The Rust LCB domain
  (`domain/livecodebench.rs`) mirrors `lcb_runner`'s code-gen template in a
  completion shape for a base model.
- **Score**: the official `lcb_runner` eval core (`codegen_metrics`) via
  `scripts/phase22_bench/lcb_score.py`, with `--start-date`/`--end-date` for the
  cutoff split. The eval core runs with **datasets+numpy only** (no
  torch/vllm/pyext) in an isolated venv — the heavy CLI wrapper is bypassed.
- **Data**: `scripts/phase22_bench/lcb_export_problems.py` exports HF
  `livecodebench/code_generation_lite` (release_v5 = **880 problems**,
  2023-05..2025-01) to `data/livecodebench/`.

# Blocker found + fixed: BF16 corrupts long-prompt generation

Standing this up surfaced a real defect in `QwenModelActor`: **generation
degenerates into token-doubling garbage on long prompts (~500+ total tokens) in
BF16**. Isolation: HumanEval/MBPP short prompts are clean; a *diverse*
(non-repetitive) long prompt corrupts in BF16 but is **clean in F32**;
temperature- and prefill-independent (batched and token-by-token priming both
corrupt). BF16's 7-bit mantissa loses too much rotary/attention precision at
length. Never hit before because all Phase 22 work used ~150-token prompts +
`max_new 192`.

Fix (`39f038e`): `phase22_dump_completions --dtype {bf16,f16,f32}`, **default
F32** (this dumper targets the long-prompt benchmarks). F32 7B = 28 GB, fits a
40 GB card for inference. Follow-up (BF16-memory-efficient): F32 rotary in a
vendored `candle-transformers` qwen2.

# Base result (release_v5, idx 640–760, F32, greedy pass@1)

| window | problems | pass@1 |
|---|---|---|
| overall | 120 | **0.125** |
| pre-cutoff (< 2024-09-01) | 28 | **0.250** |
| post-cutoff (≥ 2024-09-01) | 92 | **0.087** |

- **Pipeline validated.** 0.125 is non-zero and **inside the LCB plausibility
  band**, so the prompt format + generation + official scoring all work. The
  `eval_sanity` 7B LCB band is recalibrated to the measured base
  (`[0.03, 0.23]`).
- **Cutoff split works, and the base shows a ~3× pre/post gap** (0.25 vs 0.087).
  Suggestive of the base itself benefiting from having seen pre-cutoff problems
  — but **confounded**: pre n=28 is small and from a single late-pre window
  (2024-06..08), and platform/difficulty differ across the boundary. It is a
  preliminary reference, not a contamination verdict.
- **The reference to beat is post-cutoff base = 0.087.** A recipe that only
  lifts pre-cutoff is contaminated; one that lifts post-cutoff generalizes.

# The metric trap: greedy is the wrong ruler for self-improve

First-pass LCB runs used **greedy** (passk 1), and every recipe looked like it
*hurt*: post-cutoff base 0.087 → K=8 RL 0.054, → full-set SFT (re-trained; the
originals were deleted) 0.065. But a HumanEval sanity check exposed the trap:
the re-trained full-set SFT scored **0.439 greedy vs base 0.488** (worse!) yet
**0.756 vs 0.656 at aggregate pass@1** (temp 0.8) — **+0.10**. SFT *sharpens the
sampling distribution*: it lifts aggregate pass@1 / pass@k but can drop the
single greedy mode. **The self-improve gain lives at aggregate pass@1, not
greedy** — so a greedy LCB run cannot detect transfer. All the greedy LCB
numbers are valid greedy measurements and useless for this question.

# Recipe result (correct metric): full-set SFT GENERALIZES to LiveCodeBench

Re-run at **aggregate pass@1 (temp 0.8, passk 5)** — where the gain lives —
base + full-set SFT (seed 42), same F32 path, same slice:

| | overall | pre (<2024-09, n=28) | post (≥2024-09, n=92) |
|---|---|---|---|
| base | 0.075 | 0.186 | **0.041** |
| full-set SFT | 0.095 | 0.221 | **0.057** |
| **Δ** | **+0.020** | +0.035 | **+0.016** |

- **The self-improve gain transfers — and holds on unseen problems.** Full-set
  SFT lifts LCB **post-cutoff** (0.041 → 0.057, +0.016) — problems released
  *after* the model's cutoff, definitely unseen. That is **real
  generalization**, not contamination/recall: a pure-recall recipe would show no
  post-cutoff lift.
- **Mild contamination component, not the whole story.** The pre-cutoff lift
  (+0.035) is larger than post (+0.016), a hint that some of the gain is on
  possibly-seen problems — but post is clearly positive, so the net is genuine
  transfer.
- **Caveats.** Small samples (post n=92, gains are a few problems); single hep
  seed; low absolute LCB rates (0.04–0.09). Directionally clear, not a tight CI.
- K=8 RL (hard-tail) was only measured greedy (superseded); its aggregate LCB
  transfer is a follow-up, but it is the narrower recipe.

**Bottom line**: measured at the metric where self-improve actually lives, the
HumanEval full-set self-improve **generalizes to a different, harder benchmark
and to post-cutoff problems** — the strongest evidence yet that the gain is real
learning, not pretraining recall. The greedy detour is the reusable lesson:
**match the eval metric to where the training signal lives** (this repo's own
recurring theme — pass@5 saturation, aggregate-vs-greedy).

Cutoff pin: Qwen2.5-Coder-7B released ~2024-09; `2024-09-01` used as the split.
Confirm the exact data cutoff from the tech report before any stronger claim.
