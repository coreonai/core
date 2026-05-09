# Phase 15 S1 — HumanEval substrate qualification (HEADROOM-OK, NOISY)

Phase 14 substrate (25 single-line problems) hit a saturation
ceiling (84% saturated under SFT) that left algorithmic comparisons
no headroom — Phase 14 C4 (OPD multi-teacher) had to be deferred for
this reason. Phase 15 S1 measures the canonical HumanEval-164
substrate's variance bound and saturation distribution to decide
whether to proceed with C4-equivalent (Phase 15 S2 multi-teacher OPD).

## Setup

- **Model**: Qwen2.5-Coder-0.5B + LoRA (r=16, α=32, q_proj+v_proj)
- **Problem set**: HumanEval canonical 164 (openai/human-eval)
- **Self-improve protocol**: 1 LoRA-FT round (round-0 + final-1)
  × 3 samples × 200 train-steps
- **Generation**: max-new-tokens 200, temperature 0.8, top_p 0.95;
  truncate at top-level def/class boundary
- **Verifier**: subprocess + canonical test suite, 4s timeout
- **Seeds**: 5 (entangled init RNG + harvest sampling RNG —
  decomposition is S3a)
- **Hardware**: GPU 2 (seeds 0/1/2 sequential) + GPU 3 (seeds 3/4
  sequential), wallclock ~4.5h (each seed ~80 min — slower than
  Phase 14 S1 by ~9× since 164 vs 25 problems and longer
  generations)

## Result

Per-seed final pass rates: [0.278, 0.220, 0.224, 0.299, 0.203]

| round | mean | σ |
|---|---:|---:|
| round-0 (pre-train) | 0.213 | **0.006** |
| final-1 (after LoRA-FT) | **0.245** | **0.041** |

Per-challenge bucketing (mean across 5 seeds):
- Saturated (μ ≥ 0.95): 6 / 164 (**4%**) — Phase 14 was 84%, S1 hit
  the headroom goal
- Headroom (0.05 ≤ μ < 0.95): 86 / 164 (52%)
- Cold-start (μ < 0.05): 72 / 164 (44%)

## Substrate verdict — HEADROOM-OK but NOISY

| target | result | status |
|---|---|---|
| σ_final ≤ 0.03 | 0.041 | **MISS** (37% over) |
| saturation ≤ 50% | 4% | **HIT** (12× under) |

The substrate has plenty of movable problems, but final σ is 3.7×
Phase 14's 0.011. 2σ threshold for algorithmic comparison is **0.082**
absolute (≈ 33% relative). High bar, but actionable for genuinely
strong techniques.

## Mechanism — LoRA-FT trajectory bimodality

The σ blowup at final (0.006 → 0.041) is concentrated in *training*,
not harvest. Per-seed lift trajectory:

| seed | r0 | final-1 | Δ | LoRA-FT loss |
|---:|---:|---:|---:|---:|
| 0 | 0.211 | 0.278 | **+0.067** | 0.099 |
| 1 | 0.220 | 0.220 | +0.000 | 0.078 |
| 2 | 0.209 | 0.224 | +0.015 | (mid) |
| 3 | 0.220 | 0.299 | **+0.079** | 0.141 |
| 4 | 0.207 | 0.203 | −0.004 | 0.075 |

Group structure (lifted Δ > 0.03 vs flat |Δ| ≤ 0.03):

| group | seeds | r0-pass kept | new-pass acquired | LoRA-FT loss |
|---|---|---:|---:|---:|
| LIFTED | 0, 3 | **79%** | 31 | 0.099 / 0.141 |
| FLAT | 1, 4 | **54%** | 21 | 0.078 / 0.075 |

**Counter-intuitive overfitting signature**: FLAT seeds achieved
*lower* LoRA-FT loss (0.075-0.078) but *worse* generalization than
LIFTED seeds (0.099-0.141). Classic overfitting — high training
accuracy on idiosyncratic chosen-pair patterns destroys OOD
generalization. The forgetting (FLAT seeds lose 46% of r0
capability) is the symptom; overfitting is the cause.

Seed 2 finished with mild lift (+0.015), within noise of "flat".

## Implications for Phase 15 S2 (multi-teacher OPD)

The mechanism finding sharpens the S2 hypothesis. OPD KL-distillation
to specialist teachers is a **natural regularizer** against the exact
failure mode S1 surfaced. Two ways OPD can win:

1. **Mean shift**: Δ_mean > 0.082 (the 2σ threshold). High bar.
2. **Variance reduction**: σ_OPD < σ_SFT/2. Less ambitious — OPD
   stabilizes all student seeds to behave like LIFTED group, even if
   the mean barely moves. Still a robust algorithmic win.

S2's analyzer now reports both delta-mean AND σ-ratio so this
variance-reduction win is detectable.

## Implications for Phase 15 S3a (variance decomposition)

The lift bimodality strongly suggests **init-RNG dominates**: paired
seeds (e.g. 0/3 vs 1/4) have similar harvest sets (Jaccard 0.52-0.56,
no group difference) but radically different LoRA-FT trajectories.
S3a's σ_init / σ_harvest decomposition will quantify this. Prediction:
σ_init >> σ_harvest at HumanEval scale.

## Decision — proceed to S2 + S3a in parallel

S2 OPD goes ahead with σ=0.041 substrate. The mechanism makes OPD's
expected value clear; even if mean is flat, variance reduction is a
plausible robust win. S3a runs in parallel on idle GPUs to dissect
σ_init vs σ_harvest while S2 trains.

## Reproducing

```bash
# Setup (one-time)
mkdir -p data/humaneval
curl -fsSL https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o data/humaneval/HumanEval.jsonl.gz
gunzip data/humaneval/HumanEval.jsonl.gz

# Run 5 seeds
bash scripts/phase15_s1/run_seeds_a.sh   # GPU 2, seeds 0/1/2
bash scripts/phase15_s1/run_seeds_b.sh   # GPU 3, seeds 3/4

# Analyze
/tmp/p14_env/bin/python scripts/phase15_s1/analyze.py
```

## See also

- `docs/phase15-design.md` — full Phase 15 plan (S1-S4)
- `docs/phase14-stage-c-closeout.md` — motivation for harder
  substrate
- `scripts/phase15_s2/` — multi-teacher OPD prep (kicks off after
  this commit)
- `scripts/phase15_s3/` — init/harvest variance decomposition prep
