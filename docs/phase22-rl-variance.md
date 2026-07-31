---
title: "Phase 22 follow-up — RL variance-reduction study (pre-registration)"
date: "2026-07-31"
status: "PRE-REGISTERED — criteria locked before any test-arm eval"
---

# Why

After the C4/C5 corrections, REINFORCE on the 7B HumanEval hard tail **matches
SFT on the mean** but not on the spread:

| arm | pass@1 | pass@5 | seeds |
|---|---|---|---|
| SFT samples=16 r2 | 0.364 ± **0.037** | 0.566 ± **0.020** | 4 |
| RL posonly | 0.412 ± **0.103** | 0.581 ± **0.067** | 6 |
| base | 0.172 | 0.422 | — |

RL's mean is equal-or-better; its σ is **~2.8× wider at pass@1 (0.103 vs
0.037)** and ~3.3× at pass@5. That spread is the only reason SFT is the
deployment pick. **If the variance comes down to SFT's neighborhood without
losing the mean, RL becomes viable** — the hypothesis this study tests.

# The lever, and an important caveat from C4

The untried objective-side lever is **group-relative advantage normalization
(GRPO)**. The current RL loop already subtracts a baseline (group mean,
`v − mean(v)`); true leave-one-out (RLOO) differs only by a `k/(k−1)` rescale
for binary rewards. What is *not* yet done is dividing by the group std, which
equalizes each prompt's advantage magnitude: today a prompt passing 3/4 and one
passing 1/4 push with different-magnitude gradients; under GRPO both are
normalized to comparable scale. Optional **advantage clipping** bounds the
per-sample step.

**Caveat (C4 "Where next"):** the seed dominates the arm. Seeds 42/200 land
near 0.63 and 100/300 near 0.51 in *both* the posonly and fulladv arms, so most
of the variance is driven by **the harvest draw** (which completions get
sampled), not the objective. Phase 15 S3b reached the same conclusion for SFT
(harvest, not init, is the noise axis). So:

- Objective-side normalization (GRPO/clip) can only shrink the *within-prompt*
  contribution imbalance. It may not move the dominant harvest variance.
- The **harvest-side lever is group size K** (samples per prompt). More samples
  per prompt is the direct CLT reduction of harvest noise. This study includes
  it as a second arm rather than betting everything on the objective transform.

# Pre-registered decision criteria (LOCKED)

Fixed here, before any test-arm eval, per the standing lesson that this repo has
twice mistaken a measurement artifact for a result (C5 reward bug; the
`FilteredDomain` truncation bug). All arms are scored on **one identical ruler**.

- **Substrate:** Qwen2.5-Coder-7B + LoRA (r=16, α=32), HumanEval hard tail
  idx 100–163 (`--prompt-offset 100 --n-prompts 64`), posonly objective
  (`--pg-positive-only`), 30 RL steps, k=4, lr=2e-4, `--pg-micro-batch-size 1`,
  `--sync-every 1`, two-GPU split, bf16 train. Identical to the C4 re-run except
  for `--advantage-mode` / `--advantage-clip` / k.
- **Eval ruler:** `phase22_humaneval_baseline --offset 100 --n-problems 64
  --passk 5 --sequential --aggregate`, one binary built at HEAD, same as C4.
- **Seeds:** ≥ 6, reusing {42, 100, 200, 300, 400, 500} so each test arm pairs
  seed-for-seed against the existing MeanCenter posonly arm.
- **Primary metric:** pass@1 (aggregate). Secondary: pass@5.
- **SUCCESS** for a test arm requires **both**:
  1. **Variance:** pass@1 σ ≤ 1.5 × SFT pass@1 σ = **≤ 0.056** (from 0.103).
     Secondary check: pass@5 σ ≤ 1.5 × 0.020 = ≤ 0.030 (from 0.067).
  2. **No mean regression:** pass@1 mean ≥ SFT mean − 1σ_SFT = **≥ 0.327**
     (equivalently, not below the MeanCenter posonly mean by more than its own σ).
- **Reporting:** every arm reports mean ± σ at pass@1 and pass@5 with the SFT
  and MeanCenter-RL references, plus the paired per-seed deltas vs MeanCenter.

A test arm that lowers σ but drops the mean below 0.327, or lowers the mean-RL
gap while leaving σ ≥ 0.056, is a **failure** under these rules regardless of how
it looks on any single seed.

# Anti-illusion guards

1. **Re-measure the reward distribution first, then calibrate the clip on it.**
   The clip range must come from the *post-fix* per-prompt pass-count
   distribution under the RL sampling config (temp 0.8, k, truncation on), not
   from pre-fix data. Tuning a clip on the mis-measured (truncation-off) reward
   would re-introduce exactly the distortion C5 documented.
2. **One ruler.** All arms scored through the same evaluator/truncation path;
   no filtered-vs-unfiltered draw mixing (the `FilteredDomain` lesson).
3. **Paired seeds.** Each test arm shares seeds with the MeanCenter baseline, so
   the comparison is paired (the design that made C4's posonly>fulladv robust).
4. **Criteria frozen above** before any test-arm number is looked at.

# Arms

| arm | flag | tests |
|---|---|---|
| MeanCenter (baseline) | `--advantage-mode mean` | reproduces the existing posonly arm exactly |
| RLOO | `--advantage-mode rloo` | leave-one-out; expected ≈ MeanCenter (scale only) — a control |
| GRPO | `--advantage-mode grpo` | group-std normalization — the objective-side lever |
| GRPO+clip | `--advantage-mode grpo --advantage-clip <c>` | c from guard #1 |
| K=8 | `--k-per-prompt 8` (mode held at best of above) | harvest-side CLT lever |

RLOO is included as a **control**: if it moves σ, something other than the
std-normalization is acting (it shouldn't, for binary rewards).

# Mechanism (implemented this session)

- `qwen2_lora::AdvantageMode` {`MeanCenter`, `Rloo`, `Grpo`} + pure
  `group_advantages(verdicts, mode, clip) -> Vec<f32>`, 6 unit tests
  (mean-center = current default; RLOO = mean-center × k/(k−1); GRPO equalizes
  per-prompt magnitude; no-signal → all-zero under every mode; clip symmetric).
- `phase22_he_reinforce`: `--advantage-mode` (default `mean`, byte-identical to
  prior behavior) and `--advantage-clip`. The inline `v − mean` baseline is
  replaced by a `group_advantages` call; default path unchanged.

# Guard #1 result — base-policy reward distribution + clip decision

`scripts/phase22_rl_variance/calibrate_reward_dist.sh`, 3 seeds × step-0
(base policy frozen via `--sync-every 0`), k=4, 64 hard-tail problems,
temp 0.8 / top_k 40 / top_p 0.95 / max_new 192. Per-prompt pass-count `p`
histograms `[p0,p1,p2,p3,p4]`:

| seed | p0 | p1 | p2 | p3 | p4 | signal (0<p<4) |
|---|---|---|---|---|---|---|
| 42 | 37 | 19 | 6 | 1 | 1 | 26/64 |
| 100 | 38 | 15 | 8 | 2 | 1 | 25/64 |
| 200 | 35 | 17 | 11 | 1 | 0 | 29/64 |
| **agg (192)** | **110** | **51** | **25** | **4** | **2** | **80/192** |

Under GRPO+posonly the trained-on positive advantages are fully determined
by `p`: `p=1 → +1.732`, `p=2 → +1.0`, `p=3 → +0.577` (p=0,4 are
zero-advantage, skipped). **`p=1` — a prompt solved by a single lucky sample
out of 4 — is 64% of all signal prompts and receives the *maximum* advantage
+1.732.** That single-sample spike is exactly the high-variance term.

**Decision: `--advantage-clip 1.0`.** It caps the +1.732 (p=1) spikes to
parity with p=2's passing samples — targeting the dominant variance source —
while preserving the p2 > p3 ordering. Clip 0.577 would over-flatten (erase
the legitimate p3 signal); no clip leaves the spikes unbounded. Calibrated on
post-fix (truncation-on) data per guard #1, not pre-fix.

# Status

- [x] Mechanism + unit tests + CUDA example build (74 cudarc symbols, flags live).
- [x] Guard #1: post-fix base-policy pass-count distribution measured →
      `--advantage-clip 1.0` (p=1 dominates at 64% of signal prompts).
- [ ] **Wave 1 (running): GRPO+clip 1.0, seeds 42/100/200/300**, 30 steps,
      posonly, `--sync-every 1`. The calibration-recommended best candidate,
      tested first for the fastest decisive read.
- [ ] Score wave-1 checkpoints on the one ruler; evaluate vs criteria. Then:
      succeed → run plain GRPO (attribute norm vs clip) + extend to 6 seeds;
      fail → objective-side lever is dead (C4's harvest-driven caveat) →
      pivot to the K=8 harvest arm.
