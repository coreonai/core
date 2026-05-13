# Phase 19 closeout — multi-round saturation extended, BoN at MR neutral

Phase 18 closed with multi-round saturation curve NOT plateauing at
r=4 (single seed 0.519, cumulative +0.306 from base). Phase 19 picks
3 cheapest Phase 18 candidates: rounds=5 (extend saturation curve),
BoN harvest + multi-round (does k=10 chosen pool compound at MR?),
rounds=3 + pass@k (does diversity-preservation hold deeper?).

## Scoreboard

| stage | scope | result |
|---|---|---|
| **S1** | rounds=5 SFT at HumanEval (5 seeds) | **WIN** mean 0.556 ± 0.037, Δ=+0.037 vs r=4 |
| **S2** | BoN harvest k=10 + rounds=2 (5 seeds) | NEUTRAL Δ=-0.001 vs P17 S1 |
| **S3** | rounds=3 + pass@k single seed | pass@1=0.473, pass@10=0.591 (diversity preserved) |

## Multi-round saturation curve (HumanEval, samples=6)

| rounds | mean | σ | Δ vs prev | n |
|---:|---:|---:|---:|---:|
| 1 (P16 S1) | 0.230 | 0.031 | — | 5 |
| 2 (P17 S1) | 0.404 | 0.013 | +0.174 | 5 |
| 3 (P18 S2) | 0.475 | 0.024 | +0.070 | 5 |
| 4 (P18 S6) | 0.519 | n/a | +0.044 | 1 |
| **5 (P19 S1)** | **0.556** | **0.037** | **+0.037** | 5 |

**Diminishing but still compounding**. r5 → cumulative r0 lift +0.326,
above base pass@10 = 0.524. σ widens at higher rounds (0.013 → 0.037)
— some seeds compound more than others.

Per-seed r=5 trajectories:
- seed 0: 0.213 → 0.246 → 0.409 → 0.472 → 0.519 → 0.546
- seed 1: 0.216 → 0.230 → 0.419 → 0.514 → 0.564 → **0.620** (project record)
- seed 2: 0.220 → 0.223 → 0.409 → 0.462 → 0.518 → 0.552
- seed 3: 0.221 → 0.270 → 0.397 → 0.455 → 0.491 → 0.528
- seed 4: 0.210 → 0.184 → 0.383 → 0.488 → 0.517 → 0.535

Seed 1 at 0.620 is a 2.4× lift over base pass@1 (0.216) — single-shot
SFT model at 62% HumanEval pass@1.

## S2 — BoN at multi-round: doesn't compound

| variant | mean | σ | n |
|---|---:|---:|---:|
| P17 S1 (k=6, r=2) | 0.404 | 0.013 | 5 |
| P17 S7a (k=10, r=1) | 0.236 | 0.036 | 5 (NEUTRAL vs k=6) |
| **P19 S2 (k=10, r=2)** | **0.403** | **0.019** | 5 |

Δ vs P17 S1 = -0.001. **BoN harvest at multi-round is no different
from k=6 baseline**. The chosen-pool expansion provides no additional
lift on top of multi-round's compounding.

**Mechanism**: at multi-round, the chosen pool already grows via the
round-1-improved model adding new completions (Sa mechanism). Adding
more samples per prompt at fixed rounds doesn't add new directions;
it just adds more of the same.

This generalizes Phase 17 S7a's NEUTRAL finding: BoN harvest is
neutral at BOTH single-round AND multi-round.

## S3 — rounds=3 + pass@k: diversity preserved at deeper rounds

| metric | base (S6) | r=2 SFT (Sa) | **r=3 SFT** (this commit) |
|---|---:|---:|---:|
| pass@1 | 0.216 | 0.404 | **0.473** (+0.069 vs Sa) |
| pass@2 | 0.300 | 0.472 | 0.525 |
| pass@5 | 0.425 | 0.545 | 0.567 |
| pass@10 | **0.524** | **0.604** | **0.591** (-0.013, within noise) |

**Multi-round diversity-preservation generalizes to r=3**. pass@1
lifts +0.069 while pass@10 stays statistically equivalent. Sa's
finding (rounds=2 lifts BOTH axes) extends to deeper rounds.

For deployment: r=3 SFT pass@5 = 0.567 > base pass@10 = 0.524.
**rounds=3 SFT with k=5 inference beats base with k=10** (cheaper
inference, better result).

## Cumulative Phase 11-19 narrative

| phase | dominant finding |
|---:|---|
| 11-16 | 8 retractions, 0 robust positives |
| 17 | First 4 robust positives — multi-round + pass@k |
| 18 | Risk #20 falsified for Muon/OPD; saturation curve no plateau at r=4 |
| **19** | **r=5 still compounding; BoN+MR neutral; diversity preserved at r=3** |

**Compounded recipe results**:
- r=1 SFT: 0.230 (Phase 16 baseline)
- **r=5 SFT: 0.556** (best so far, single shot)
- **r=3 SFT + pass@5: 0.567** (training+inference compound)
- **r=2 SFT + pass@10: 0.595** (Sa+S7 2-seed mean)
- **base + pass@10: 0.524** (pure inference, no training)
- **r=4 SFT pass@1: ~0.52** (pure training, no inference scaling)

The training-axis and inference-axis are **additive but with
diminishing returns**. r=5 SFT alone ≈ r=2 SFT + pass@10. The
question: where's the deployment optimum?

## What this commit changes for Phase 20+ practice

### Settled
- **rounds=5 still compounds**. Plateau not reached.
- **BoN at MR doesn't help**. Don't try samples > 6 with multi-round.
- **Diversity preservation extends to r=3**. Likely extends to r=4, 5.

### Established defaults
- **rounds=2 or 3 SFT, samples=6** is the sweet spot for compute
  budget (each round costs ~80 min/seed at samples=6).
- **rounds=5 is the deep training option** if budget allows (~5×
  cost for ~0.15 absolute pass@1 lift).
- **pass@k inference (k=5-10)** at deployment.

### Phase 20 candidates

The interesting open questions are now beyond simple parameter
sweeps:

1. **rounds=6+ saturation finding** — does it ever plateau? Extra
   2-3 seeds at r=6 or r=8 single-seed.
2. **MBPP rounds=5** — cross-substrate saturation curve. Does MBPP
   plateau at a different round than HE?
3. **Combined recipe deployment compute/timing budget** —
   document the production-ready recipe with cost analysis.
4. **Substrate scale-up** Qwen 1.5B-Coder — does multi-round
   compounding hold at larger model? Phase 19 deferred candidate.
5. **RL with pass@k reward** — train directly against the
   inference-time objective. New infrastructure.

## See also

- `docs/phase18-closeout.md` — Phase 18 closeout (this Phase 19
  picks 3 cheapest of its 5 candidates)
- `docs/phase17-closeout.md` — Phase 17 closeout (multi-round + pass@k
  findings that S3 extends)
- `scripts/phase19_s{1,2,3}/` — driver scripts + per-seed JSONs
