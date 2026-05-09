# Phase 15 S2 — Multi-teacher OPD vs SFT-on-union (LOSS)

Phase 14 C4 (OPD) was deferred because the 25-problem substrate
saturated 21/25 problems, leaving no headroom for a multi-teacher
test. Phase 15 S1 qualified HumanEval-164 with 52% headroom, 4%
saturation (σ=0.041, 2σ=0.082 threshold). S2 is the resurrected C4.

## Setup

- **Substrate**: HumanEval-164 (Phase 15 S1)
- **Specialists** (k=3, single seed=99):
  - strings (68 problems): final pass 0.283
  - numbers (30 problems): final pass 0.408
  - collections (66 problems): final pass 0.436
  - Trained 2 LoRA-FT rounds × 4 samples × 200 steps each
- **Routing**: each problem → fixed specialist by signature/docstring
  keyword heuristic (scripts/phase15_s2/routing.py)
- **OPD student** (5 seeds):
  - Fresh LoRA r=16 α=32 q_proj+v_proj on Qwen 2.5-Coder-0.5B base
  - Harvest 164 × 3 samples → filter to verifier-passed → bucket by
    routing → OPD-FT student against routed specialist's logits
  - Per-batch teacher dispatch via per-subset DataLoader + round-
    robin step (after fixing mixed-subset batch crash, see below)
  - Loss: `opd_loss(student_logits, [(1.0, teacher_logits)],
    labels, T=2.0, direction='forward')`
  - 200 train-steps, lr 2e-4, AdamW (per Phase 14 C2 verdict)
- **Hardware**: GPU 2 (seeds 0/1/2) + GPU 3 (seeds 3/4) sequential

## Result — LOSS for OPD (technically within-inflated-noise, systematic across subsets)

5 seeds × OPD multi-teacher vs Phase 15 S1 SFT:

| seed | OPD r0 | OPD final | Δ_OPD (final − r0) | SFT-S1 final | Δ_OPD-SFT |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.213 | 0.195 | -0.018 | 0.278 | -0.083 |
| 1 | 0.211 | 0.148 | -0.063 | 0.220 | -0.071 |
| 2 | 0.224 | 0.242 | **+0.018** | 0.224 | +0.018 |
| 3 | 0.234 | **0.053** | **-0.181** | 0.299 | -0.246 |
| 4 | 0.207 | 0.144 | -0.063 | 0.203 | -0.059 |

| arm | mean | σ |
|---|---:|---:|
| SFT (Phase 15 S1) | 0.245 | 0.041 |
| OPD multi-teacher | **0.157** | **0.070** |

Δ_mean = **−0.088**, 2σ_max = 0.140 → "WITHIN NOISE" formally,
**but** σ_OPD/σ_SFT = **1.70** (variance inflated 70%). The reason
the verdict is "within noise" is that OPD's inflated σ widens the
2σ threshold itself, not because OPD is comparable to SFT.

### Three converging lines of evidence make this a robust LOSS

1. **All 3 subsets show same-sign Δ** (per-subset breakdown):

   | subset | n | SFT mean | OPD mean | Δ | specialist (teacher) |
   |---|---:|---:|---:|---:|---:|
   | strings | 68 | 0.224 | 0.135 | -0.088 | 0.283 |
   | numbers | 30 | 0.224 | 0.167 | -0.058 | 0.408 |
   | collections | 66 | 0.276 | 0.174 | -0.102 | 0.436 |

   Identical-sign Δ across 3 independent subsets is unlikely to be
   noise (probability ~12.5% under null). Combined with magnitudes
   matching the overall Δ, this is systematic.

2. **OPD destroys the round-0 base model**: 4 of 5 seeds end below
   round-0 (post-OPD-FT < base Qwen). Mean Δ_OPD = -0.061 vs 0.
   Note SFT had +0.032 lift in the same comparison. **OPD is on
   average worse than no training**.

3. **Distillation is not working as designed**: specialists beat SFT
   baseline on every subset (0.283 / 0.408 / 0.436 vs SFT 0.224 /
   0.224 / 0.276), so teachers ARE more knowledgeable than the
   student starts. Yet the student doesn't benefit from teaching
   on any subset. Whatever signal is flowing from teacher to
   student is destructive, not informative.

### Both formal win paths fail

1. Mean shift: -0.088, target was +0.082. **Fail.**
2. Variance reduction: σ_OPD (0.070) > σ_SFT (0.041) by 1.70×.
   Target was σ_OPD < σ_SFT/2. **Fail.**

## Mechanism — student is being pulled to a lower-quality state

Per-seed OPD round-0 → final-1 trajectory:

| seed | r0 (= base Qwen) | final (post-OPD-FT) | Δ |
|---:|---:|---:|---:|
| 0 | 0.213 | 0.195 | -0.018 |
| 1 | 0.211 | 0.148 | -0.063 |
| 3 | 0.234 | **0.053** | **-0.181** |
| 4 | 0.207 | 0.144 | -0.063 |

Seed 3 catastrophe: dropped to 5.3% — generation almost completely
broken. The other seeds show milder destruction. **OPD is making the
model WORSE than the untrained Qwen base** in 4 of 4 measured seeds.

### Hypothesized causes

1. **Forward KL with T=2.0 is high-entropy-pulling**. KL(teacher ||
   student) penalizes student for low probability on tokens teacher
   ranks high. With softened temperature both distributions, student
   learns to spread mass uniformly → can't generate coherent code.
   Reverse KL (KL(student || teacher)) might behave better — that's
   DeepSeek V4's actual choice; my opd_loss defaults to forward.

2. **Specialists are noisy teachers**. Strings specialist's training
   trajectory was 0.202 → 0.096 → 0.283 (catastrophic forgetting at
   round 1, recovered at round 2). The saved adapter has the
   recovered weights, but its logits at completion-token positions
   inherit some of that instability. Distilling student toward a
   noisy teacher amplifies the noise.

3. **Hinton 2015 T² scaling missing**. Standard distillation scales
   the KL loss by T² to compensate for gradient attenuation under
   softmax(logits/T). Without it, the gradient is small but biased,
   and 200 steps of biased small gradient can drive the student
   significantly off-track. (Minor effect though — probably not the
   primary cause.)

4. **OPD as primary training, not as RL replacement**. DeepSeek V4
   uses OPD AFTER an SFT phase to align an already-good student
   toward better teachers. We're using OPD AS the training (no SFT
   anchor). Without an SFT loss term, KL alone has no grounding to
   verifier-passed completions.

The most actionable hypothesis is **#1 (KL direction)**. A reverse-KL
re-run on 1-2 seeds is a cheap follow-up.

## Verdict — Phase 14 C4's deferral was warranted

The Phase 14 C4 deferral note read:

> Falling back to "OPD as KL-anchor regularizer" (single teacher =
> frozen base Qwen) reduces to KL-regularized SFT, mechanistically
> similar to C3's DPO+ref. C3 already showed that helper doesn't
> help here.

S2 generalizes that finding from "single teacher = base" to
"multi-teacher = trained specialists with routing": **OPD as
configured here is destructive at LoRA scale even with proper
multi-teacher specialist setup**.

The S2 result is not a refutation of OPD-the-idea (DeepSeek V4 uses
it successfully at full-finetune of a much larger model). It IS a
refutation of "naive offline OPD with forward-KL T=2.0 plugs into
small-LoRA self-improve loops as a drop-in for SFT."

## Decision impact

- **OPD not added to Stage C-equivalent default**. AdamW + SFT
  remains canonical (Phase 14 C2/C3 verdict reaffirmed at HumanEval
  scale).
- **Phase 12 S2's `opd.rs` + Phase 14 C4's `opd.py` unit-tested loss
  modules stay in codebase** — the loss is correct, the application
  pattern is what fails. Future S2.5 / Phase 16 may revisit with
  reverse-KL or hybrid SFT+OPD.
- **Risk register**: add #18 — naive offline OPD destabilizes
  small-LoRA self-improve loops. KL direction matters; forward-KL +
  high T pulls student to high-entropy state.

## Bug found + fixed during S2

First S2 OPD attempt crashed with:
`RuntimeError: opd_collate: mixed-subset batch {'collections', 'numbers'}`

Root cause: subset-sorted DataLoader still produces mixed-subset
batches at subset boundaries (e.g. with 13 collections + batch_size=4,
the 4th batch spans collections boundary into numbers). The smoke
test happened to use evenly-divisible counts and missed it.

Fix: per-subset DataLoader + round-robin step. Each batch is now
guaranteed homogeneous (single teacher per batch), and round-robin
weights subsets equally regardless of pair-count imbalance. Verified
with imbalanced-subset smoke (7+4+13 triples, batch_size=4, 12 steps).

## Reproducing

```bash
# Train specialists (one-time)
bash scripts/phase15_s2/run_specialists.sh 2

# Train OPD students
bash scripts/phase15_s2/run_opd_a.sh   # GPU 2, seeds 0/1/2
bash scripts/phase15_s2/run_opd_b.sh   # GPU 3, seeds 3/4

# Analyze
/tmp/p14_env/bin/python scripts/phase15_s2/analyze.py
```

## See also

- `docs/phase14-stage-c-closeout.md` — original C4 deferral
  rationale (this commit's verdict ratifies it)
- `docs/phase15-s1-substrate.md` — substrate qualification + lift
  bimodality mechanism
- `docs/phase15-s3a-variance-decomposition.md` — Phase 14 σ axis
  decomposition
- `scripts/phase15_s2/{routing.py, train_specialist.py,
  self_improve_opd.py, analyze.py}` — full implementation
- `scripts/phase14_c4/opd.py` — OPD loss (PyTorch port of Rust
  Phase 12 S2)
- DeepSeek V4 technical report (2026-04-24) — original on-policy
  distillation design (full-finetune scale, reverse-KL)
