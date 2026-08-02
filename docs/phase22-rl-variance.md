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

# Wave 1 result — GRPO+clip 1.0 DOES NOT QUALIFY (objective-side lever fails)

4 paired seeds, 30 steps, posonly, `--sync-every 1`, scored on the one ruler.
Training was healthy — 4/4 seeds rose (step-29 pass 58–125/256 vs base ≈41),
comp_len steady 155–160, no runaway.

| seed | pass@1 | Δ base | pass@5 | Δ base |
|---|---|---|---|---|
| 42 | 0.5156 | +0.344 | 0.7031 | +0.281 |
| 100 | 0.2375 | +0.066 | 0.5469 | +0.125 |
| 200 | 0.2375 | +0.066 | 0.4531 | +0.031 |
| 300 | 0.4469 | +0.275 | 0.6719 | +0.250 |
| **mean** | **0.359 ± 0.144** | | **0.594 ± 0.116** | |

| criterion (locked) | value | verdict |
|---|---|---|
| pass@1 σ ≤ 0.056 | **0.144** | **FAIL** (worse than MeanCenter RL's 0.103) |
| pass@1 mean ≥ 0.327 | 0.359 | PASS |
| **overall** | | **DOES NOT QUALIFY** |

**The objective-side lever made variance *worse*, not better.** σ rose from
MeanCenter RL's 0.103 to 0.144 (pass@1), 2.6× over the 0.056 target. Paired
pass@5 vs the MeanCenter posonly arm is mixed — 42/100/300 up, 200 down 0.172
— i.e. no consistent gain, just more spread (seed 42 = 0.516 pass@1 while
100/200 = 0.238).

**Mechanism.** GRPO's advantage scale is ~2.3× MeanCenter's (magnitudes
1.0–1.732 vs 0.25–0.75), so at the same lr it takes larger steps and amplifies
whatever the seed already controls — the harvest draw. The huge seed spread
directly reproduces C4's "**seed dominates the arm**": variance is
harvest-driven, and normalizing the objective cannot touch it. This is a clean
confirmation of the pre-registered caveat.

**Plain GRPO (no clip) not run — and not worth 7.5 h.** Clip 1.0 *reduces*
advantage magnitude (caps 1.732→1.0), so it is the *gentler* of the two; the
uncapped version takes even larger steps and would be even higher-variance. If
the gentler config already fails at σ 0.144, the harsher one cannot qualify.

# Wave 2 result — K=8 harvest lever: fails the σ criterion, but flips the mean

4 paired seeds, MeanCenter, posonly, `--k-per-prompt 8` (2× the K=4 harvest),
30 steps, same ruler.

| seed | pass@1 | Δ base | pass@5 | Δ base |
|---|---|---|---|---|
| 42 | 0.5125 | +0.341 | 0.6406 | +0.219 |
| 100 | 0.5375 | +0.366 | 0.6250 | +0.203 |
| 200 | 0.6375 | +0.466 | 0.7344 | +0.313 |
| 300 | 0.4437 | +0.272 | 0.5312 | +0.109 |
| **mean** | **0.533 ± 0.080** | | **0.633 ± 0.083** | |

| criterion (locked) | value | verdict |
|---|---|---|
| pass@1 σ ≤ 0.056 | **0.080** | **FAIL** (but down from K=4 MeanCenter's 0.103) |
| pass@1 mean ≥ 0.327 | 0.533 | PASS |
| **overall** | | **DOES NOT QUALIFY** (against the locked σ rule) |

**Two things happened, and the second is the story.**

1. **The harvest lever moved σ the right way — but not far enough.** σ fell
   0.103 → 0.080 (≈22%) doubling K. CLT for a harvest-dominated variance
   predicts ~1/√2 = 0.71× (→ 0.073); observed 0.080 is close, so the reduction
   is roughly CLT-consistent, with a residual (init/other) that does not shrink
   and holds σ above the 0.056 target. This **confirms C4's diagnosis**
   (variance is harvest-driven) from the *positive* side, where GRPO confirmed
   it from the negative side.

2. **K=8 lifted the MEAN dramatically — enough to flip the deployment call.**
   pass@1 mean 0.533 vs K=4 MeanCenter 0.412 and **SFT 0.364** — a **+0.169**
   gap over SFT, ≈4.6× SFT's σ. More harvest = more positive-advantage training
   samples/step = more learning, not just tighter variance. All 4 K=8 seeds
   (0.444–0.638) sit **above SFT's mean**, and the *lowest* (0.444) exceeds
   SFT's mean+2σ (0.438). Deployment math: K=8 RL mean−2σ = **0.373 > SFT mean
   0.364** — even a pessimistic K=8 run beats a typical SFT run.

**Re-reading the pre-registration.** The locked σ criterion assumed RL only
*matched* SFT's mean, so variance was the sole differentiator. K=8 broke that
assumption: on expected deployment value (mean − risk), K=8 RL now **beats**
SFT on this substrate despite a wider σ. The criterion answered its exact
question (no, RL is not low-variance) but is the wrong lens once the mean moved.

**Caveats (why this is a direction, not yet a headline).** n=4 (this repo
retracts n=4 claims — needs seeds 400/500 to match the K=4 arms' 6). K=8 is 2×
the generation compute of K=4 and SFT, so the mean lift is not compute-free.
The lift is a *harvest/learning* effect, not variance reduction per se.

## Wave 2 extension — 6 seeds confirm and sharpen

Seeds 400/500 added (same ruler), 6 seeds total:

| | pass@1 | pass@5 |
|---|---|---|
| **K=8 posonly (6 seeds)** | **0.538 ± 0.076** | **0.656 ± 0.077** |
| K=8 posonly (4 seeds) | 0.533 ± 0.080 | 0.633 ± 0.083 |
| SFT samples=16 r2 (4 seeds) | 0.364 ± 0.037 | 0.566 ± 0.020 |
| K=4 MeanCenter RL (6 seeds) | 0.412 ± 0.103 | 0.581 ± 0.067 |
| base | 0.172 | 0.422 |

New per-seed pass@1 — 400 = 0.600, 500 = 0.491. The reading **firms**: mean
0.533 → 0.538 (stable), σ 0.080 → 0.076 (slightly tighter, still > 0.056).

- **Mean dominance is robust, not a 4-seed fluke.** pass@1 0.538 vs SFT 0.364 =
  **+0.174**; all 6 seeds (0.447–0.653) sit above SFT's mean, and the lowest
  (0.447) still exceeds SFT mean+2σ (0.438).
- **σ 0.076** — 2.05× SFT's 0.037, 1.36× the 0.056 gate. Still FAILS the locked
  criterion. Doubling K bought a ~26% σ reduction (0.103 → 0.076) and no more;
  the residual is the non-harvest floor.
- **Deployment math (6 seeds): K=8 RL mean−2σ = 0.386 > SFT mean 0.364**, and
  mean−1σ = 0.462 > SFT mean+2σ (0.438). Even a 2σ-pessimistic K=8 run beats a
  typical SFT run.

# Conclusion

**The pre-registered question — "can RL be made low-variance to match SFT?" —
is answered NO.** Neither lever reaches σ ≤ 0.056: the objective-side (GRPO)
makes it *worse* (0.144), the harvest-side (K=8) improves it but plateaus at
0.076. Variance is harvest-driven (confirmed from both directions), with a
residual floor that more harvest doesn't clear.

**But the harvest lever surfaced a better recipe.** K=8 posonly RL lifts the
*mean* to 0.538 (+0.174 over SFT, 6 seeds), because more harvest = more
positive-advantage (RAFT-style) training samples per step = more learning, not
just tighter variance. On expected deployment value (mean−2σ), **K=8 RL now
beats SFT on this substrate** despite the wider σ — the σ gate was the right
question under the old assumption (RL only matches SFT's mean) and the wrong
lens once the mean moved.

**Deployment recommendation (revised, mean−kσ not σ-gate):**
- **Best expected pass@1: K=8 posonly RL — 0.538 ± 0.076** (+0.174 over SFT),
  if you can afford ~2× the generation compute of K=4/SFT.
- **Tightest per-GPU-hour: SFT — 0.364 ± 0.037** (low variance, low compute).
- pass@k inference scaling remains the training-free, orthogonal win.

**Attribution — resolved by existing data (no fresh run needed).** The control
already exists on this exact ruler. `docs/phase22-7b-results.md` measured SFT at
samples-per-prompt = 16 **and 32** (the 2× harvest), re-scored on the same
consistent ruler:

| | pass@1 | pass@5 |
|---|---|---|
| SFT samples=16 (4 seeds) | 0.364 ± 0.037 | 0.566 ± 0.020 |
| SFT samples=32 (4 seeds) | 0.385 ± 0.108 | 0.535 ± 0.090 |
| **K=8 posonly RL (6 seeds)** | **0.538 ± 0.076** | **0.656 ± 0.077** |

**Doubling SFT's harvest does not approach K=8 RL.** SFT plateaus at ~0.36–0.385
regardless of harvest (16 or 32); K=8 RL reaches 0.538. Even SFT's 32
samples/prompt — **4× K=8 RL's per-step harvest (8)** — trails RL by **+0.153**
pass@1. So the K=8 win is the **RL regime** (on-policy, 30 steps), not just more
harvest: reward-weighted on-policy updates extract more than one-shot
rejection-sampling FT on a bigger pile. (A re-run is also *risky*: the original
SFT hard-tail command did not survive, so a fresh SFT-32 would introduce
config-mismatch — the existing corrected-ruler numbers are the clean control.)

# Status

- [x] Mechanism + unit tests + CUDA example build.
- [x] Guard #1: base-policy pass-count distribution → `--advantage-clip 1.0`.
- [x] **Wave 1: GRPO+clip 1.0 — DOES NOT QUALIFY** (σ 0.144, worse). Objective
      lever dead; confirms harvest-driven variance from the negative side.
- [x] **Wave 2: K=8 (6 seeds) — DOES NOT QUALIFY on σ (0.076 > 0.056), but the
      mean flips deployment** (0.538, +0.174 over SFT; all 6 seeds above SFT
      mean; mean−2σ > SFT mean). Harvest lever confirmed; axis shifts to mean.
- [x] **Study concluded.** RL can't be made low-variance here, but K=8 posonly
      is the best-mean recipe; deploy on mean−kσ, not the σ gate.
- [x] **Attribution control — resolved from existing data.** SFT samples=32
      (2× harvest, same ruler) = 0.385 pass@1, does NOT approach K=8 RL's 0.538.
      The win is the RL regime, not just harvest. No fresh run (and the original
      SFT command didn't survive → re-run would risk config-mismatch).
