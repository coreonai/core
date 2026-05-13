# Phase 20 design — saturation closure + cross-substrate + deployment recipe

Phase 19 closed with multi-round saturation curve still NOT plateauing at
r=5 (mean 0.556 ± 0.037, Δ=+0.037 vs r=4). Phase 19 also confirmed BoN+MR
neutral and diversity preservation at r=3. Phase 20 picks the 3 cheapest
Phase 19 candidates to close the loop on the saturation finding:

| stage | scope | hypothesis |
|---|---|---|
| **S1** | rounds=6 SFT at HumanEval (5 seeds) | does compounding finally plateau at r=6? |
| **S2** | rounds=5 SFT at MBPP (5 seeds) | does HumanEval saturation curve generalize cross-substrate? |
| **S3** | deployment compute/timing budget doc (no GPU) | production-ready recipe with cost analysis |

All three are independent. S1/S2 are GPU-bound; S3 is docs.

## S1 — rounds=6 SFT saturation closure

**Setup**: `scripts/phase15_s1/self_improve.py` with `--rounds 6 --samples 6
--train-steps 200`. Same protocol as P17/P18/P19 SFT runs, just one more
round.

**Decision gates** (vs P19 S1 r=5 mean 0.556 ± 0.037):

- r=6 mean > 0.590: monotonic compounding continues (Δ +0.034 > σ)
- 0.55 < r=6 mean < 0.59: plateau begins (Δ within 1σ)
- r=6 mean < 0.55: saturation reached, r=5 is the sweet spot
- r=6 mean drops below r=5: overshoot/overfitting

Per-round Δ history target table:
- r1 +0.03 → r2 +0.17 → r3 +0.07 → r4 +0.04 (n=1) → r5 +0.037 → r6 ???

**Cost per seed**: 7 harvest rounds × ~80 min = ~9.3h sequential.
5 seeds × 1 GPU each in parallel = ~9.3h wallclock.

## S2 — MBPP rounds=5 cross-substrate

**Setup**: `scripts/phase17_sb/run_mr_mbpp.py` with `--rounds 5 --samples 6
--train-steps 200`. Phase 18 S3 already did MBPP rounds=3 (mean 0.457).
Phase 20 extends to r=5 to test if HumanEval saturation generalizes.

**Decision gates** (vs P18 S3 MBPP r=3 mean 0.457 ± 0.013, and predicted
saturation curve from HE):

- MBPP r=5 mean > 0.55: cross-substrate saturation confirmed (parallel curves)
- 0.50 < mean < 0.55: positive but flatter curve than HE
- 0.46 < mean < 0.50: minimal additional compounding (MBPP saturates earlier)
- mean ≤ 0.457: MBPP fully saturates at r=3 (cross-substrate divergence)

**Cost per seed**: 6 harvest rounds × ~80 min = ~8h sequential.
5 seeds × 1 GPU each in parallel = ~8h wallclock. Run AFTER S1 finishes
(GPU contention).

## S3 — Deployment compute/timing budget (no GPU)

**Deliverable**: `docs/phase20-deployment-recipe.md` capturing:

- Training cost per recipe (wall-clock × GPU-hours)
- Inference cost per query (tokens × pass@k samples)
- Pareto front: pass-rate vs total compute
- Recommended recipes for 3 budgets (cheap/balanced/max)
- Production checklist: checkpointing, eval reproducibility, drift monitoring

Pulls numbers from Phase 17-19 closeouts. No new measurement.

## Hardware allocation

| GPU pair | stage | seeds | wallclock | start |
|---|---|---|---:|---|
| 0+1+5+6+7 | S1 rounds=6 | 5 | ~9.3h | Wave 1 (now) |
| 0+1+5+6+7 | S2 MBPP r=5 | 5 | ~8h | Wave 2 (after S1) |
| (no GPU) | S3 docs | — | ~1h | Wave 1 parallel |

Total wallclock: ~18h sequential. (Could parallelize S1+S2 by splitting
GPUs 3+2 but slower per seed.)

## What Phase 20 does NOT test (deferred to Phase 21+)

- **rounds=8 single-seed** — defer until r=6 result lands. If r=6 already
  plateaus, r=8 unnecessary; if r=6 still compounds, r=8 is next.
- **Qwen 1.5B-Coder substrate** — Phase 19 deferred, still high-cost.
- **RL with pass@k reward** — requires new infra (days of code work).
- **Pekko actor integration** — Phase 21+ infrastructure work.

## See also

- `docs/phase19-closeout.md` — saturation curve r=1..5
- `docs/phase18-closeout.md` — saturation curve r=1..4 + cross-substrate
- `docs/phase17-closeout.md` — first robust positives (MR + pass@k)
- `scripts/phase20_s{1,2}/` — driver scripts + per-seed JSONs
- `docs/phase20-deployment-recipe.md` — S3 deliverable (this commit)
