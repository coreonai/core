# Phase 15 closeout — substrate qualified, 4 retractions, 1 substrate-shape lesson

Phase 15 set out to (a) qualify a harder substrate than Phase 14's
saturated 25-problem set, then (b) test the deferred Phase 14 C4
(multi-teacher OPD) and decompose σ axis-by-axis. 4 stages
completed, 1 substrate qualification, 3 algorithmic retractions, 1
σ-decomposition mechanism finding.

## Scoreboard

| stage | scope | result | commit |
|---|---|---|---|
| S1 | HumanEval substrate qualification | 0.245 ± 0.041, 4% saturated, 52% headroom | c4f4ed7 |
| S2 | Multi-teacher OPD vs SFT | **LOSS** Δ=−0.088 (systematic across subsets) | 2bdc8c5 |
| S3a | Phase 14 σ axis decomposition | σ_harvest 93%, σ_init 7% | 1df7820 |
| S3b | HumanEval σ axis decomposition | σ_harvest 97%, σ_init 3% | 2bdc8c5 |
| S4 | Muon at HumanEval | **LOSS** Δ=−0.081 (5/5 seeds, generalizes C2) | this commit |

## Cumulative Phase 14 + 15 retraction count

| technique | source paper | tested in | Δ | failure mechanism |
|---|---|---|---:|---|
| Muon (NS-orthogonalized SGD-momentum) | DeepSeek V4 | C2, S4 | -0.092, -0.081 | Wrong inductive bias for rank-r LoRA (substrate-independent) |
| DPO variants (hybrid, round-0-only) | Phase 11 S5 | C3 | within noise | Pair scarcity at saturating substrates |
| OPD (multi-teacher offline distillation) | DeepSeek V4 | S2 | -0.088 | KL direction + noisy teachers, destabilizes LoRA student |

**3 of 3 DeepSeek V4 techniques fail at LoRA self-improve scale.**
Three distinct failure mechanisms:
- Optimizer (Muon) — orthogonalization conflicts with low-rank bottleneck
- Preference (DPO) — needs failure mass that LoRA quickly drains
- Distillation (OPD) — KL pulls student to high-entropy state

The "hot 2026 paper drop" testing strategy at this scale:
**high-yield in failure-mode discovery, low-yield in net gains**.

## Phase 15 substrate is harvest-RNG-dominated (both substrates)

S3a + S3b together establish a substrate-shape-independent finding:

| substrate | σ_init | σ_harvest | init share |
|---|---:|---:|---:|
| Phase 14 (saturated 25) | 0.004 | 0.016 | 7% |
| Phase 15 (HumanEval 164) | 0.009 | 0.050 | 3% |

S1 mechanism analysis predicted σ_init >> σ_harvest at HumanEval
(LIFTED/FLAT seed bimodality). S3b refuted this — Jaccard measures
*which problems pass*, but harvest-RNG also controls *which
completions* are generated. Same passing-set, different completions,
different LoRA-FT trajectory.

**Actionable**: multi-init averaging is useless (3% of variance);
**multi-sample averaging IS the noise-reduction lever**. Phase 15 S1
used samples=3 → σ=0.041. Doubling to samples=6 should roughly
halve σ_harvest by CLT. For σ ≤ 0.030 target: samples=6. For σ ≤
0.020: samples=12.

## What Phase 15 leaves the project with

### Improved infrastructure

- **HumanEval 164 substrate harness**: `scripts/phase15_s1/`
  with multi-line generation, top-level-def truncation, subprocess
  test-suite verification. Reusable for any future Qwen-LoRA
  algorithmic comparison.
- **Multi-teacher OPD pipeline (PEFT 4-model)**: `scripts/phase15_s2/`
  with subset routing, per-subset DataLoader, round-robin step,
  `opd_loss` PyTorch port (also at `scripts/phase14_c4/`).
- **Variance-axis decomposition harness**: `scripts/phase15_s3/`
  with separate --init-seed / --harvest-seed flags for both Phase 14
  and HumanEval substrates.
- **Muon-vs-AdamW at HumanEval scale**: `scripts/phase15_s4/`
  reusing `scripts/phase14_c2/muon.py`.

### Risk register additions

- **Risk #16 generalized**: optimizer transfer (full-finetune → LoRA)
  is non-monotonic. Confirmed across substrate shapes via Phase 14 C2
  + Phase 15 S4.
- **Risk #18 (new)**: naive offline OPD destabilizes small-LoRA
  self-improve loops. KL direction matters; forward-KL + high
  temperature pulls student to high-entropy state.

### Refuted predictions

- S1 mechanism prediction (σ_init >> σ_harvest at HumanEval because
  LIFTED/FLAT had similar Jaccard) — refuted by S3b. Explanation:
  Jaccard measures pass-set, not completion-set.

## What's pending — Phase 16 candidates

1. **Reverse-KL OPD re-run**. Single-seed quick test. If it works
   substantially better than forward-KL (S2 verdict), Phase 14
   C4-equivalent is salvageable. If not, OPD is closed at LoRA
   scale.
2. **Hybrid OPD + SFT**. α-mixing OPD KL with SFT NLL anchor (mirror
   of Phase 11 S5 hybrid DPO). May reduce KL's destructive pull.
3. **samples=6 substrate re-run**. Empirically validate S3b
   prediction that doubling samples halves σ. If σ drops to ~0.025
   as predicted, all subsequent algorithmic comparisons get tighter
   2σ thresholds (~0.05) and weak signals become detectable.
4. **A different model family**. Qwen2.5-Coder-0.5B saturates +
   destabilizes around the same scale across both substrates. Try
   OLMo-1B or a different Qwen variant to see if mechanisms are
   model-specific.
5. **A different LoRA rank**. r=16 may be the bottleneck for Muon's
   step-magnitude problem. r=64 or r=128 could reveal whether
   higher rank shifts Muon from LOSS to neutral or win.

The natural next phase is **Phase 16 — substrate-noise reduction +
selective S4-style re-tests** at samples=6, with one or two of the
above as algorithmic candidates.

## What to NOT spend more cycles on

- Naive Muon at LoRA — definitively retracted across substrates.
- Pure forward-KL OPD at small LoRA — S2 verdict robust, mechanism
  understood.
- Multi-init averaging — only 3% of variance.
- DeepSeek V4 directly-ported recipes at our scale — 3/3 fail; need
  scale or recipe modifications first.

## See also

- `docs/phase14-stage-c-closeout.md` — Phase 14 closeout (motivation
  for Phase 15)
- `docs/phase15-design.md` — original Phase 15 plan
- `docs/phase15-s1-substrate.md` — S1 result + mechanism (refuted)
- `docs/phase15-s2-opd-results.md` — S2 OPD LOSS
- `docs/phase15-s3a-variance-decomposition.md` — Phase 14 σ axis
- `docs/phase15-s3b-humaneval-decomposition.md` — HumanEval σ axis
- `docs/phase15-s4-muon-humaneval.md` — S4 Muon LOSS
