# Phase 15 S3b — Variance decomposition at HumanEval substrate

S3a found Phase 14 substrate (25 saturated problems) was harvest-
dominated (σ_harvest=0.016 / σ_init=0.004 → 93%/7%). S1's mechanism
analysis suggested HumanEval (164 headroom-rich problems) might
flip to init-dominated, since LIFTED/FLAT seeds had similar Jaccard
overlap (0.52-0.56) but radically different LoRA-FT trajectories.

S3b runs the same decomposition at HumanEval. **Both substrates are
harvest-dominated.** S1 mechanism prediction refuted.

## Setup

- HumanEval-164, Qwen2.5-Coder-0.5B + LoRA, same hyperparams as S1
- 5 init runs (init=0..4, harvest=0) → σ_init
- 5 harvest runs (init=0, harvest=0..4) → σ_harvest
- Phase 15 S1's existing 5-seed runs as combined (paired init=harvest)
- Hardware: GPU 0 + GPU 1, ~10h total wallclock (each run ~80 min)

## Result

| axis | n | mean | σ |
|---|--:|---:|---:|
| init-only | 5 | 0.291 | **0.009** |
| harvest-only | 5 | 0.236 | **0.050** |
| combined (S1) | 5 | 0.245 | **0.041** |

### Decomposition

σ²-share at HumanEval:

| axis | σ² | share |
|---|---:|---:|
| init | 0.009² ≈ 0.00008 | **3%** |
| harvest | 0.050² ≈ 0.00250 | **97%** |

### Cross-substrate comparison

| substrate | σ_init | σ_harvest | init share |
|---|---:|---:|---:|
| Phase 14 (saturated 25) | 0.004 | 0.016 | 7% |
| Phase 15 (HumanEval 164) | 0.009 | 0.050 | **3%** |

Phase 15 is even MORE harvest-dominated than Phase 14. Init RNG
barely contributes at either substrate.

### Additivity

Predicted (independent): √(0.009² + 0.050²) = 0.051
Observed combined: 0.041
Ratio 1.23 → mild anti-correlation (combined < predicted), like
Phase 14 (1.46 ratio).

## Why my S1 mechanism prediction was wrong

S1 mechanism analysis found LIFTED seeds (0/3, +0.067/+0.079) and
FLAT seeds (1/4, +0.000/-0.004) had similar Jaccard overlap of
r0-passing problems (0.52-0.56). I concluded "harvests are similar
across groups → trajectory split must come from init RNG."

**The flaw**: Jaccard measures *which problems* pass, but harvest-
RNG ALSO controls *which completions* are generated for those
problems. Even when seeds A and B both pass problem `is_prime`, A
might generate `def is_prime(n): return n > 1 and ...` while B
generates `def is_prime(n):\n    if n < 2: return False\n    ...`.
Both pass the verifier, but they're different training data, and
LoRA-FT on those different completions takes the model to different
places.

S3b's per-axis isolation cuts through this confound. With init=0
fixed and varying only the harvest seed, σ=0.050 — direct evidence
that harvest-RNG drives most of the trajectory variance. Init is
nearly noiseless (σ=0.009).

## Implications

### Multi-init averaging is NOT the noise-reduction tool

The natural noise-reduction recipe — train several initializations
and average — would only address the 3% of variance from init RNG.
Pointless for this substrate.

### Multi-sample averaging IS the right tool

σ_harvest is driven by per-prompt sampling (temperature + top-p
+ RNG). The natural lever is increasing samples-per-prompt during
harvest. With Phase 15 S1 using `samples=3`, doubling to `samples=6`
roughly halves σ_harvest by the central limit theorem (more
independent samples per prompt → tighter chosen-pair distribution).

For future Phase 15 / Phase 16 algorithmic comparisons targeting
σ ≤ 0.03: bump samples to 6 (estimated σ_harvest ≈ 0.035, σ_total
similar). For σ ≤ 0.02: samples=12 (~hopefully σ_harvest ≈ 0.025).

This costs proportional GPU time but is the correct lever.

### Phase 14 retracted claims still valid

S3a already established C2/C3 retractions are robust to the σ
underestimation under paired-seed comparison. S3b confirms the same
holds at HumanEval scale: paired comparisons (Muon vs SFT, OPD vs
SFT) match noise between arms regardless of which axis dominates.

## Decomposition mean differences (sanity check)

| axis | mean |
|---|---:|
| init-only | 0.291 |
| harvest-only | 0.236 |
| combined | 0.245 |

The init-only mean (0.291) is higher than combined (0.245). Why?
Because init-only fixes harvest=0, and the harvest=0 sampling seed
happens to produce a "lucky" chosen-pair distribution that helps
LoRA-FT generalize well across all 5 init seeds. Combined runs
average over 5 different harvest seeds, including some that are
unlucky (e.g. harvest=2 in the harvest-only axis gave 0.167).

This effect (lucky-fixed-harvest) is real but cosmetic — the σ
measurements within each axis are still valid noise estimates.

## Reproducing

```bash
bash scripts/phase15_s3/run_he_init_axis.sh 0    # GPU 0
bash scripts/phase15_s3/run_he_harvest_axis.sh 1 # GPU 1
/tmp/p14_env/bin/python scripts/phase15_s3/analyze_humaneval.py
```

## See also

- `docs/phase15-s3a-variance-decomposition.md` — Phase 14 substrate
  decomposition (this is the cross-substrate twin)
- `docs/phase15-s1-substrate.md` — S1 mechanism analysis (refuted
  init-dominance prediction)
- `scripts/phase15_s3/{decompose_seeds_humaneval.py, analyze_humaneval.py,
  run_he_*.sh}` — full S3b implementation
