# Phase 16 Design — substrate noise reduction + 2 mechanism re-tests

Phase 15 closed with 3/3 DeepSeek V4 techniques retracted at LoRA
self-improve scale, plus a substrate-shape-independent finding
(both substrates harvest-RNG-dominated). Phase 16 picks the highest-
leverage follow-ups from the closeout's "Phase 16 candidates" list:

1. **S1 — samples=6 substrate** (validates σ halving prediction)
2. **S2 — Reverse-KL OPD** (DeepSeek V4's actual KL direction; tests
   if Phase 15 S2 LOSS was specifically about forward-KL)
3. **S3 — Muon at LoRA r=64** (tests if higher rank fixes Phase 14
   C2 + Phase 15 S4's "wrong inductive bias for rank-r" mechanism)

All three run in parallel on disjoint GPU pairs.

## S1 — samples=6 substrate qualification

**Goal**: validate S3b's CLT prediction that doubling samples-per-
prompt halves σ_harvest. Phase 15 S1 used samples=3 → σ=0.041.
Predicted samples=6 → σ ≈ 0.029 (target ≤ 0.030 satisfied).

**Setup**:
- HumanEval-164, Qwen2.5-Coder-0.5B + LoRA r=16 α=32, 1 round LoRA-FT,
  AdamW lr=2e-4, train-steps=200
- `samples=6` (vs Phase 15 S1's samples=3)
- 5 seeds × ~160 min/seed (2× harvest cost) ÷ 2 GPUs = ~7h

**Decision gate**:
- σ ∈ [0.025, 0.035]: S3b prediction validated. samples=6 becomes
  default for Phase 16+ algorithmic comparisons.
- σ < 0.025: better than predicted (CLT under-estimates because
  per-prompt samples are correlated). Even better.
- σ > 0.040: prediction fails. Need different noise-reduction
  approach (longer training, multi-checkpoint, etc.).

## S2 — Reverse-KL OPD

**Goal**: test if Phase 15 S2's LOSS (Δ=-0.088, σ blowup 1.70×) was
specifically about forward-KL direction. DeepSeek V4 used REVERSE-KL.

**Setup**:
- Same as Phase 15 S2 (multi-teacher OPD, k=3 specialists, T=2.0)
- `--kl-direction reverse` instead of `forward`
- 5 seeds × ~80 min/seed ÷ 2 GPUs = ~3.5h
- Reuses existing specialists in `checkpoints/phase15_s2/`

**Decision gate** (using Phase 15 S1 σ=0.041):
- Δ_OPD-SFT > +0.082 (2σ): Robust WIN. Phase 15 S2 verdict was
  KL-direction-specific; OPD viable at LoRA scale with reverse-KL.
  Phase 14 C4 partially salvaged.
- |Δ| ≤ 0.082: Within noise. Reverse-KL doesn't help meaningfully.
- Δ < −0.082: Same robust LOSS as forward-KL. OPD application
  pattern fails regardless of KL direction at this scale.

## S3 — Muon at LoRA r=64

**Goal**: test if Phase 14 C2 + Phase 15 S4's "NS orthogonalization
removes step-magnitude info that rank-r LoRA needs" mechanism is
rank-specific. r=16 was the bottleneck in both prior tests; r=64
gives 4× capacity. If Muon still LOSES, mechanism is more
fundamental than rank limitation.

**Setup**:
- Same as Phase 15 S4 (HumanEval-164, Qwen + LoRA, 1 round, 200 steps)
- `--lora-r 64 --lora-alpha 128` (preserves α/r=2 ratio)
- 5 seeds × Muon (AdamW r=64 baseline derived from same seeds)
- ~80 min/seed ÷ 2 GPUs = ~3.5h

**Decision gate**:
- Δ_Muon-AdamW(r=64) > +0.082: Robust WIN. Rank limitation was the
  problem; Muon viable at higher rank. Phase 14 C2 / Phase 15 S4
  retroactively scoped to "Muon at LoRA r ≤ 16 LOSES".
- |Δ| ≤ 0.082: Within noise.
- Δ < −0.082: Mechanism is rank-independent. Muon definitively
  doesn't help LoRA at any practical rank.

Note: S3 also produces a useful AdamW r=64 vs AdamW r=16 baseline
delta — does higher rank help SFT itself at this substrate?

## Parallelization plan

| GPU pair | stage | seeds | wallclock |
|---|---|---|---:|
| 0 + 1 | S1 samples=6 | 5 | ~7h |
| 2 + 3 | S2 reverse-KL OPD | 5 | ~3.5h |
| 5 + 6 | S3 Muon r=64 (+ AdamW r=64 baseline) | 5+5 | ~7h |

Total wallclock: ~7h (bottleneck = S1 / S3 6-pair). All three
analyzers ready at end. Closeout commits decisions.

## Code reuse

- S1: `scripts/phase15_s1/self_improve.py` with `--samples 6`. New
  driver scripts only.
- S2: `scripts/phase15_s2/self_improve_opd.py` with
  `--kl-direction reverse`. New driver scripts only.
- S3: `scripts/phase15_s4/run_muon.py` with `--lora-r 64 --lora-alpha 128`.
  Plus a parallel AdamW r=64 baseline (5 more runs).

Minimal new code; mostly thin run-script wrappers + new analyzers
that compare against the right baselines.

## See also

- `docs/phase15-closeout.md` — Phase 15 summary + Phase 16 candidate
  list (this design picks #1, #2, #4 from that list)
- `docs/phase15-s2-opd-results.md` — Phase 15 S2 LOSS (S2 here is
  the reverse-KL re-test)
- `docs/phase15-s4-muon-humaneval.md` — Phase 15 S4 LOSS (S3 here
  is the higher-rank re-test)
- `docs/phase15-s3b-humaneval-decomposition.md` — S3b's σ halving
  prediction (S1 here is the validation)
