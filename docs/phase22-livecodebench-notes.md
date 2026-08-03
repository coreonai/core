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

# Recipe result: K=8 RL does NOT transfer to LiveCodeBench

Same F32 Rust path, same slice, K=8 RL (the best-mean HumanEval recipe), 2 seeds
(the full-set SFT checkpoints were deleted, so only the hard-tail recipes
survive). Reference: base post-cutoff 0.087.

| | overall | pre (<2024-09, n=28) | post (≥2024-09, n=92) |
|---|---|---|---|
| base | 0.125 | 0.250 | **0.087** |
| K=8 RL seed 200 | 0.108 | 0.250 | 0.065 |
| K=8 RL seed 42 | 0.092 | 0.250 | 0.043 |
| **K=8 RL mean** | 0.100 | 0.250 | **0.054** |

- **The hard-tail K=8 RL recipe does not transfer to LCB — it slightly *hurts*.**
  post-cutoff base 0.087 → RL 0.054; overall 0.125 → 0.100. The HumanEval
  hard-tail self-improve (idx 100–163, 64 problems) is a **narrow
  specialization**, not general code-gen ability, so it degrades a different,
  harder benchmark. Expected direction; now measured.
- **No contamination signature.** The recipe leaves pre-cutoff *identical*
  (0.250 = 7/28 for base and both seeds) and only moves post-cutoff — the
  opposite of a "helps-pre-not-post" contamination pattern. So this run finds
  no contamination; it finds **no transfer**.
- **Caveats.** K=8 RL is narrow by construction (a HumanEval subset), so this is
  not a test of "does self-improve generalize" in general — the fairer test is a
  full-set recipe, whose checkpoints were deleted. Small samples (post n=92, pre
  n=28). The base's own pre/post gap (0.25 vs 0.087) is the more interesting
  observation but is confounded (difficulty across the boundary).

**Bottom line**: the self-improve gain is **benchmark-specific** — it buys
HumanEval, not competitive programming. To test broad generalization/
contamination properly, re-train a full-set recipe and re-run this path.

Cutoff pin: Qwen2.5-Coder-7B released ~2024-09; `2024-09-01` used as the split.
Confirm the exact data cutoff from the tech report before any stronger claim.
