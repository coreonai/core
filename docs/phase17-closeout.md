# Phase 17 closeout — 6 phases of training-axis failures, then 3 robust positives

Phases 11-16 spent ~30 GPU-days on training-side interventions
(Muon, OPD variants, DPO variants, hybrids, label smoothing) at
small-LoRA self-improve scale. Cumulative: 8 retractions, 0 robust
algorithmic wins. Phase 17 picked the 4 highest-leverage Phase 16
candidates (multi-round, label smoothing, MBPP cross-substrate,
inference-time techniques) plus 4 follow-up stages that exploited
early findings. **3 of the 11 Phase 17 stages produced robust
positives** — the first algorithmic wins in 6 phases.

## Scoreboard

| stage | scope | result |
|---|---|---|
| **S1** | Multi-round SFT (rounds=2 samples=6) at HumanEval | **✓ WIN** mean 0.405 ± 0.013 (Δ=+0.174 vs single-round, 2.8× threshold) |
| S2 | Label smoothing α=0.1 (overfitting regularizer) | **FAIL** mean 0.181 ± 0.062 (Δ=-0.049, 4/5 seeds negative) |
| S3 | MBPP-100 single-round SFT (cross-substrate) | qualified — mean 0.363 ± 0.024 (Δ=+0.146 vs base, ~4× HE's lift) |
| **S6** | pass@k at base Qwen (inference-time scaling) | **✓ WIN** pass@10=0.524 vs pass@1=0.216, Δ=+0.308 (5× threshold) |
| S7a | samples=10 SFT (chosen-pool expansion) | within noise — Δ_mean=+0.003 σ_blowup 1.3× (preliminary 4/5) |
| S7b | Single-round SFT pass@k eval | confirms diversity collapse — SFT pass@10=0.494 (-0.030 vs base) |
| S8 | pass@20 HumanEval (saturation check) | pass@20=0.585, diminishing returns past k=10 |
| S9 | pass@10 MBPP (cross-substrate inference) | Δ=+0.270 — S6 finding generalizes |
| **SA** | Multi-round + pass@k eval (mechanism test) | **✓ WIN** MR-SFT pass@1=0.404 + pass@10=0.604 (both > base!) |
| SB | MBPP multi-round (cross-substrate) | mean 0.453 ± 0.016 (Δ=+0.093 vs single-round MBPP) |

## 3 robust positives (each 2σ+, distinct mechanisms)

### Win #1 — S6: Inference-time scaling unlocks +0.308

Base Qwen-Coder-0.5B at HumanEval:
- pass@1 = 0.216
- pass@10 = 0.524 (+0.308 unbiased estimate)
- pass@20 = 0.585 (saturating; +0.06 marginal)

Cross-substrate (S9 MBPP): Δ pass@10-pass@1 = +0.270 — generalizes.

**Mechanism**: Pretraining gave the model multi-modal solution
distribution; T=0.8 sampling explores it; verifier picks correct
samples. **The model already "knows" the answer ~50-60% of the time
on these benchmarks; we just need to let it try multiple times.**

### Win #2 — S1: Multi-round SFT compounds (training-axis)

| substrate | single-round mean | multi-round (r2) mean | Δ |
|---|---:|---:|---:|
| HumanEval (S1) | 0.230 | **0.405 ± 0.013** | +0.174 |
| MBPP (SB) | 0.361 | **0.453 ± 0.016** | +0.093 |

Both substrates show compound lift at round 2. **σ also reduces**
(HE: 0.031 → 0.013, 0.43× ratio) — multi-round both lifts mean AND
tightens variance.

**Mechanism**: Round-1 chosen pool comes from base Qwen (~21% pass).
Round-2 chosen pool comes from round-1-trained model (~25-27% pass).
The round-2 chosen pool is strictly broader AND includes "newly
learnable" problems. Compound effect: each round expands the
training distribution along new axes rather than sharpening the same
axis.

### Win #3 — SA: Multi-round preserves pass@k (mechanism finding)

| | pass@1 | pass@10 |
|---|---:|---:|
| base | 0.216 | 0.524 |
| single-round SFT | 0.234 | 0.494 (-0.030 vs base) |
| **multi-round SFT** | **0.404** | **0.604** (+0.080 vs base) |

This is the **mechanism resolution** of S7b's "training collapses
pass@k" finding. Single-round SFT does collapse pass@k by sharpening
into a narrow distribution. But multi-round SFT **broadens** the
distribution along new chosen-pair directions each round → pass@k
RISES, not falls.

**Deployment math**:
- Multi-round SFT pass@5 = 0.545 ≈ base pass@10 = 0.524
- → **MR-SFT with k=5 matches base with k=10** (50% inference compute savings)

## Phase 17 retraction (1 stage)

### S2: Label smoothing α=0.1 fails as overfitting regularizer

5-seed result: LS 0.181 ± 0.062 vs SFT 0.230 ± 0.031. Δ=-0.049,
4/5 seeds negative, σ blew up 2×.

Phase 15 S1 mechanism analysis identified overfitting as the cause
of lift bimodality. Label smoothing is a textbook overfitting
regularizer — but at this LoRA self-improve scale, it instead
**increases diversity in the wrong direction**: forces probability
mass onto every vocab token uniformly, destroying useful code-token
modes.

Mechanism-targeted interventions are not automatic wins. The
mechanism diagnosis (S1) was correct; the proposed remedy (S2) was
wrong direction.

## Cumulative Phase 11-17 ledger

| technique | tested in | result |
|---|---|---|
| Muon (NS-orthogonalized SGD-momentum) | C2 / P15 S4 / P16 S3 | LOSS × 3 (substrate × rank) |
| DPO variants | C3 / P11 S5 | within noise × 2 |
| OPD (forward/reverse-KL, hybrid) | P15 S2 / P16 S2 / P16 S4 | LOSS × 3 |
| Label smoothing α=0.1 | **P17 S2** | LOSS (this commit) |
| Multi-init averaging | P15 S3a / P16 S1 | ineffective (3-7% of variance) |
| Higher LoRA rank (r=64) | P16 S3 | ineffective (σ blowup) |
| **Multi-round SFT** | **P17 S1, SA, SB** | **WIN × 3** (HE + cross-substrate + preserves pass@k) |
| **Inference-time pass@k** | **P17 S6, S8, S9** | **WIN × 3** (HE + saturation + MBPP) |
| **samples=6 substrate noise reduction** | P16 S1 | qualified (CLT validated) |

**4 phases of training-axis nulls, then training intervention
(multi-round) and inference intervention (pass@k) both win at this
scale.** Mechanism: previous training interventions sharpened the
narrow base distribution; multi-round broadens iteratively, and
pass@k exploits the breadth directly.

## Decision impact

### What becomes default

- **Multi-round SFT (rounds=2)** is the new default training protocol
  at HumanEval and MBPP. samples=6 was already default from P16.
- **pass@k inference at deployment** if k=5-10 budget available.
  Multi-round SFT preserves the inference advantage.
- **Combined recipe** (multi-round train + pass@k deploy): ~0.6
  pass-rate on HumanEval-164, ~0.5 on MBPP-100.

### What's settled (don't try again)

- Naive single-axis training interventions (label smoothing, samples
  scaling, Muon, OPD, DPO, higher rank) at this LoRA self-improve
  scale — all retracted across multiple variants.
- Multi-init averaging — established to capture < 10% of variance.

### What's open for Phase 18

1. **rounds=3+ self-improve** — does S1 compounding continue?
   Diminishing returns at r=3? Or new lift?
2. **Multi-round + best-of-n training-time harvesting** — combine
   S1's multi-round with k=10 harvest. Hypothesis: more
   chosen-pair diversity per round = even bigger compounding.
3. **Inference-time training (RL with pass@k reward)** — directly
   train against the pass@k objective rather than chosen-only SFT.
4. **Larger model substrate** — Qwen 1.5B/3B-Coder. The mechanism
   findings should be model-scale-independent but worth confirming.
5. **Multi-round at saturating ceiling** — when does multi-round
   plateau? Phase 17 measured r=2 only.

## See also

- `docs/phase16-closeout.md` — Phase 16 closeout with Phase 17
  candidates list (this commit completes #1, #2, #3, #4)
- `docs/phase17-s6-passk-base.md` — S6 inference-time scaling
  detailed writeup
- `docs/phase17-s1-multi-round.md` — S1 multi-round detailed
  writeup (preliminary; updated by this closeout)
- `scripts/phase17_s{1,2,3,6,7,8,9,a,b}/` — all stage scripts +
  per-seed JSON results

## Improved infrastructure carried forward

- `scripts/phase17_s1/` — multi-round driver (just `--rounds 2`)
- `scripts/phase17_s3/{problems.py, run_mbpp.py}` — MBPP-100 harness
- `scripts/phase17_s6/run_passk.py` — pass@k eval at any model
- `scripts/phase17_sa/run_mr_passk.py` — combined train + pass@k
  eval (the mechanism test format)
- `scripts/phase17_sb/run_mr_mbpp.py` — MBPP multi-round
- Cumulative: HumanEval and MBPP substrates fully covered with
  baseline + multi-round + pass@k variants
