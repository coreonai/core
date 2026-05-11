# Phase 18 design — three multi-round follow-ups

Phase 17 closed with 4 robust wins (multi-round SFT × 3 + pass@k inference × 1)
plus Risk #20: rounds=1 single-round is diversity-collapsing, so
Phase 14-16 algorithm retractions (Muon, OPD, DPO) under rounds=1
may NOT transfer to multi-round. Most actionable revisit: Muon at
multi-round LoRA.

Phase 18 picks the 3 highest-leverage Phase 17 candidates from the
closeout doc:

| stage | scope | hypothesis being tested |
|---|---|---|
| S1 | Multi-round Muon at HumanEval (rounds=2 samples=6) | Risk #20: did rounds=1 hide Muon? |
| S2 | rounds=3 SFT at HumanEval | does S1 compounding continue or plateau? |
| S3 | MBPP MR seeds 2, 4 completion | tighten σ for SB cross-substrate |

All three run in parallel on disjoint GPU pairs. ~15h wallclock for
S2 (longest); ~10h for S1; ~4h for S3.

## S1 — Multi-round Muon at HumanEval

**Setup**: Reuses `scripts/phase15_s4/run_muon.py` with `--rounds 2
--samples 6 --optimizer muon`. Same LoRA hyperparams as Phase 17 S1
SFT baseline (r=16 α=32, AdamW for SFT, Muon for this variant).

**Decision gate** (Phase 17 multi-round AdamW SFT mean = 0.404):
- Muon MR mean > 0.404 + 2σ ≈ 0.43 → ROBUST WIN, Phase 14/15/16
  retractions reinstated as rounds=1-specific
- Muon MR mean ≈ AdamW MR (within ±0.062) → Muon ≈ AdamW at
  multi-round; the 6 retractions stand as rounds=1 verdicts
- Muon MR mean < 0.404 - 2σ → Muon truly LOSES even at multi-round;
  mechanism rank-AND-round independent

This is **the** most actionable Phase 17 follow-up. Cheapest possible
revision of 3 prior retractions.

## S2 — rounds=3 SFT

**Setup**: Reuses `scripts/phase15_s1/self_improve.py` with `--rounds 3
--samples 6`. Same protocol as Phase 17 S1 but one more round.

**Decision gate** (vs Phase 17 S1 rounds=2 mean 0.404):
- rounds=3 mean > 0.466 (= 0.404 + 0.062) → compounding continues
- 0.45 < mean < 0.466 → mild compounding, near plateau
- mean ≤ 0.404 + ε → plateau at rounds=2 (rounds=2 is sweet spot)
- mean < 0.404 → catastrophic forgetting at round 3

This determines whether multi-round is monotonic improvement or has
a sweet spot.

## S3 — MBPP multi-round 5-seed completion

**Setup**: SB had seeds 0, 1, 3 done (3-seed σ=0.016). S3 here adds
seeds 2, 4 for full 5-seed σ estimate.

**Decision gate**: SB 3-seed mean = 0.453 σ=0.016. Expected 5-seed
mean ~0.45 ± 0.02. Statistical confirmation only.

## Hardware

- GPU 0+1: S1 multi-round Muon (5 seeds)
- GPU 2+3: S2 rounds=3 SFT (5 seeds)
- GPU 5:   S3 MBPP MR seeds 2, 4 (2 seeds, sequential)
- GPU 4:   busy with another user (37GB)
- GPU 6, 7: idle reserve

## What Phase 18 does NOT test

- More OPD variants — Phase 16 S2/S4 + Phase 15 S2 retracted forward,
  reverse, hybrid. Diminishing return on more KL hyperparams.
- DPO variants — Phase 14 C3 retracted hybrid + round-0 at HumanEval.
- Higher LoRA rank — Phase 16 S3 found r=64 hurts variance.
- Substrate scale-up — Qwen 1.5B-Coder substrate change is high-cost
  next phase, deferred until after Phase 18 multi-round verdicts
  land.

## See also

- `docs/phase17-closeout.md` — Phase 17 summary + Phase 18 candidate
  list (this design picks #1, #3, #5)
- `docs/phase17-s1-multi-round.md` — S1 multi-round HumanEval
- `docs/phase17-s6-passk-base.md` — S6 inference-time scaling
- `scripts/phase18_s{1,2,3}/` — driver scripts
