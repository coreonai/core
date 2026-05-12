# Phase 19 design — saturation + compound positive recipes

Phase 18 closed with Risk #20 falsified (Muon, OPD still LOSS at
multi-round) and multi-round saturation curve NOT plateauing at r=4
(rounds=4 single seed = 0.519, cumulative +0.306). Phase 19 picks
the 2 highest-leverage Phase 18 candidates:

| stage | scope | hypothesis being tested |
|---|---|---|
| **S1** | rounds=5 SFT at HumanEval (5 seeds) | does compounding plateau at r=5? |
| **S2** | Best-of-N (samples=10) + rounds=2 at HE (5 seeds) | does BoN harvest compound with multi-round? |

Both target HumanEval substrate to align with Phase 17/18 baselines.

## S1 — rounds=5 SFT saturation finding

**Setup**: Reuses `scripts/phase15_s1/self_improve.py` with `--rounds 5
--samples 6 --train-steps 200`. Same protocol as Phase 17 S1 +
Phase 18 S2 + Phase 18 S6 but with one more round.

**Decision gate** (vs Phase 17 S1 rounds=2 mean 0.404, Phase 18 S2
rounds=3 mean 0.475, Phase 18 S6 rounds=4 single seed 0.519):

- rounds=5 mean > 0.55: monotonic compounding continues
- 0.50 < rounds=5 mean < 0.55: diminishing returns plateau
- mean ≤ 0.50: saturation hit; rounds=4 is the sweet spot

Per-round Δ history:
- r1: ~+0.03
- r2: ~+0.17 (compounding kick)
- r3: ~+0.07
- r4: ~+0.04 (single seed)
- r5: ???

Each additional round costs ~80 min/seed harvest + ~30s LoRA-FT.
5 seeds × 6 harvests/seed × 80 min ÷ 2 GPUs = ~20h wallclock.

## S2 — Best-of-N harvest at multi-round

**Setup**: Reuses `scripts/phase15_s1/self_improve.py` with `--rounds 2
--samples 10` (vs Phase 17 S1's `--samples 6`).

**Decision gate** (vs Phase 17 S1 mean 0.404 ± 0.013 + Phase 17 S7a
mean 0.236 ± 0.036 single-round samples=10 NEUTRAL):

- BoN+MR mean > 0.466 (= 0.404 + 0.062): robust positive lift
  from chosen-pool expansion at multi-round
- 0.40 < mean < 0.466: within noise of r=2 baseline (BoN harvest
  doesn't compound with MR)
- mean < 0.40: chosen-pool expansion HURTS at MR (overfitting-prone)

Per-seed cost: 3 harvest rounds × 130 min (samples=10) = ~6.5h/seed.
5 seeds ÷ 2 GPUs = ~16h wallclock.

## Hardware allocation

| GPU pair | stage | seeds | wallclock |
|---|---|---|---:|
| 0 + 1 | S1 rounds=5 | 5 | ~20h |
| 5 + 6 | S2 BoN+MR | 5 | ~16h |
| 2, 3, 4 | busy with another user | — | — |
| 7 | idle reserve | — | — |

Total wallclock: ~20h (S1 bottleneck).

## What Phase 19 does NOT test (deferred)

- **Combined recipe deployment compute/timing budget** — narrative
  documentation work; defer until S1+S2 results define the final
  recipe.
- **Substrate scale-up (Qwen 1.5B-Coder)** — high cost (~2× wallclock),
  defer to Phase 20 if S1/S2 land positive.
- **RL with pass@k reward** — requires new infrastructure (~days
  of code work).
- **MBPP rounds=5** — secondary cross-substrate confirmation; can
  add if HE rounds=5 shows interesting pattern.

## See also

- `docs/phase18-closeout.md` — Phase 18 closeout (this design picks
  candidates #2 and #3)
- `docs/phase17-closeout.md` — Phase 17 closeout (multi-round +
  pass@k findings that S1/S2 extend)
- `scripts/phase19_s{1,2}/` — driver scripts
