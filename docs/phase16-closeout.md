# Phase 16 closeout — substrate noise lever validated, OPD definitively closed

Phase 16 ran 4 stages targeting Phase 15's "Phase 16 candidates":
substrate noise reduction (S1), reverse-KL OPD (S2), Muon at higher
LoRA rank (S3), and hybrid OPD+SFT (S4 added during execution).

## Scoreboard

| stage | scope | result | commit |
|---|---|---|---|
| S1 | samples=6 substrate (CLT validation) | σ 0.041→0.031, ratio 0.75 (CLT 0.71) ✓ | this |
| S2 | Reverse-KL OPD | LOSS Δ=−0.159 (worse than forward-KL!) | cb87f03 |
| S3 | Muon at LoRA r=64 | LOSS Δ=−0.070 (rank-independent) | cb87f03 |
| S4 | Hybrid OPD+SFT α=0.3 | LOSS Δ=−0.114, σ blowup 2.1× | this |

## Key findings

### S1: substrate noise model validated

Phase 15 S3b's harvest-RNG-dominated finding + CLT prediction:
σ ∝ 1/√samples. samples=3→6 predicted σ_ratio = 0.71. **Observed
0.75 — within 6% of theory**. Confirms harvest-RNG is the dominant
noise axis at both substrates. Multi-init averaging was a wrong
turn; samples-per-prompt is the right lever.

New 2σ threshold for Phase 16+: **0.062** (down from 0.083). All
future algorithmic comparisons can adopt samples=6 to detect weaker
algorithmic deltas.

### S2: KL direction doesn't fix OPD — actually makes it worse

| variant | mean | σ | Δ vs SFT |
|---|---:|---:|---:|
| Forward-KL OPD (P15 S2) | 0.157 | 0.070 | −0.088 |
| Reverse-KL OPD (P16 S2) | **0.086** | 0.045 | **−0.159** |

Reverse-KL is 2× more destructive than forward-KL. Mechanism:
reverse-KL's mode-seeking on noisy specialist teachers concentrates
student into teacher's worst modes. Forward-KL's mode-spreading is
merely destructive; reverse-KL is pathological.

### S3: Muon mechanism is rank-independent

| substrate | rank | Δ Muon-AdamW |
|---|---|---:|
| Phase 14 (saturated 25) | r=16 | -0.092 |
| Phase 15 (HE 164) | r=16 | -0.081 |
| Phase 15 (HE 164) | r=64 | **-0.070** |

Three substrate × rank combinations all LOSS for Muon. Updated
mechanism: NS strips per-direction magnitude info; AdamW's per-
parameter scaling fits the ~100-300-pair training-data scale.

Side-finding: AdamW r=64 σ blows up 2.4× (0.041→0.097) due to one
catastrophic seed. Higher rank = overfitting-prone at this data
scale. Phase 16+ stays at r=16 even for AdamW. Risk #19 added.

### S4: Hybrid OPD+SFT partially rescues, still LOSS

| variant | mean | σ |
|---|---:|---:|
| SFT only | 0.245 | 0.041 |
| Reverse-KL OPD pure | 0.086 | 0.045 |
| **Hybrid α=0.3** | **0.130** | **0.088** |

Hybrid moves mean +0.044 over pure reverse-KL (real rescue effect)
but σ blows up to 0.088 (2.1× σ_SFT). Trimodal distribution: 2
SFT-level seeds, 1 mid, 2 catastrophic. Mirror of Phase 14 C3
hybrid DPO pattern. SFT anchor at α=0.3 prevents some catastrophes
but never produces positive lift.

## Cumulative Phase 14 + 15 + 16 retraction count

| technique | tested in | result | mechanism |
|---|---|---|---|
| Muon (NS-orthogonalized SGD-momentum) | C2, P15 S4, P16 S3 | LOSS × 3 | Wrong inductive bias for LoRA at this data scale; rank-independent |
| DPO variants (hybrid α=0.3, round-0-only) | C3 | within noise | Pair scarcity at saturating substrates |
| OPD (forward-KL, reverse-KL, hybrid) | P15 S2, P16 S2, P16 S4 | LOSS × 3 | KL alone has no chosen-pair anchor; specialist-quality-dependent destabilization |

**3/3 DeepSeek V4 techniques retracted across multiple
hyperparameter and substrate configurations**:
- Muon: 3 substrate × rank combinations
- DPO: 11-variant matrix at K9 + 2 variants at HumanEval
- OPD: 3 KL/anchor variants at HumanEval

The "hot 2026 paper drop" testing strategy at our LoRA-self-improve
scale has now produced ~7 algorithmic retractions across Phases 11-16
with no robust algorithmic win. The right inference is NOT "the
techniques are wrong" — DeepSeek V4 uses them productively at
full-finetune scale — but rather that **naive ports of full-finetune-
scale techniques to small-LoRA self-improve don't transfer**.

## Phase 17 candidates

The path forward should NOT be more direct paper ports. Instead:

1. **Substrate scale-up**: Qwen 1.5B-Coder or Qwen 3B-Instruct as
   base. Phase 9 S4 tested 1.5B-Coder with mixed results (sum-AUC
   degraded, F=8 lift dropped) — but for self-improve, larger model
   may give Muon/OPD enough capacity to express their inductive
   biases.
2. **Multi-step self-improve depth**: all our experiments use 1
   round (round-0 + final-1). Phase 4 work on multi-round dynamics
   used K9 substrate; Phase 16+ multi-round at HumanEval is
   untested. Some techniques (e.g. EWC for forgetting) may shine
   only at multi-round.
3. **Algorithmic primitives, not paper ports**: e.g. test
   "label smoothing as overfitting regularizer" (cheap, addresses
   Phase 15 S1's overfitting mechanism directly).
4. **Cross-axis variance audit completion**: temperature axis +
   checkpoint axis still untested. May reveal that some single-run
   "robust" effects are noise.

## What NOT to spend cycles on

- **More OPD variants** — 3 retractions, mechanism converging.
  Don't try OPD on this LoRA-self-improve scale. Move up to bigger
  model OR change to fine-tuning-from-pretrained-student paradigm.
- **More Muon variants** — 3 retractions across substrate × rank.
  Mechanism converging on data-scale not rank.
- **More DPO at LoRA-saturating substrates** — pair scarcity is
  fundamental.
- **Multi-init averaging** — confirmed 3-7% of variance.

## Improved infrastructure carried forward

- `scripts/phase16_s1/` — samples=6 substrate harness. Can run any
  algorithmic test at tighter σ.
- `scripts/phase16_s2/` — reverse-KL OPD harness (just `--kl-direction`
  flag in P15 S2's trainer).
- `scripts/phase16_s3/` — Muon-vs-AdamW at any LoRA rank (just
  `--lora-r` flag).
- `scripts/phase16_s4/` — hybrid OPD+SFT trainer with `--sft-alpha`.
  Reusable for any future hybrid-distillation experiment.

## See also

- `docs/phase15-closeout.md` — Phase 15 closeout (motivation +
  Phase 16 candidate list)
- `docs/phase16-design.md` — Phase 16 plan
- `docs/phase16-s1-samples6-substrate.md` — S1 σ validation
- `docs/phase16-s2-reverse-kl-opd.md` — S2 OPD KL-direction LOSS
- `docs/phase16-s3-muon-r64.md` — S3 Muon rank-independent LOSS
- `docs/phase16-s4-hybrid-opd-sft.md` — S4 hybrid LOSS
