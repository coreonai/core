# Phase 18 closeout — Risk #20 verdicts + multi-round saturation curve

Phase 17 closed with 4 robust wins (multi-round SFT × 3 + pass@k
inference × 1) plus Risk #20: rounds=1 may have been the dominant
factor in Phase 11-16's 8 retractions. Phase 18 tests Risk #20 for
the two cheapest revisable retractions (Muon, OPD) and maps the
multi-round saturation curve (r=3, r=4, MBPP). 7 stages total.

## Scoreboard

| stage | scope | result |
|---|---|---|
| **S1** | Multi-round Muon at HumanEval (5 seeds) | **LOSS** Δ=-0.148 |
| **S2** | rounds=3 SFT at HumanEval (5 seeds) | **WIN** Δ=+0.070 vs r2 |
| **S3** | MBPP multi-round completion (2 seeds → P17 5-seed) | mean 0.457 ± 0.013 |
| **S4** | Multi-round Reverse-KL OPD (5 seeds) | **LOSS** Δ=-0.145 |
| **S5** | Multi-round SFT + pass@k MBPP (2 seeds) | cross-substrate diversity preserved |
| **S6** | rounds=4 single seed (compounding saturation) | 0.519 (no plateau yet) |
| **S7** | Sa replication seed 1 (HumanEval pass@k) | Sa 2-seed mean pass@1=0.415, pass@10=0.595 |

## Risk #20 verdicts — falsified for both Muon and OPD

Phase 17 closeout's hypothesis: rounds=1 single-round was the
diversity-collapsing protocol that may have hidden Muon/OPD positives
in Phase 11-16. Phase 18 S1 + S4 directly test by running both at
rounds=2.

### S1 — Multi-round Muon LOSES across ranks × rounds × substrates

| substrate | rank | rounds | Δ Muon-AdamW | seeds Muon<SFT |
|---|---|---|---:|---:|
| Phase 14 (sat 25) | r=16 | 1 | -0.092 | 5/5 |
| Phase 15 (HE 164) | r=16 | 1 | -0.081 | 5/5 |
| Phase 16 (HE 164) | r=64 | 1 | -0.070 | 4/5 |
| **Phase 18 (HE 164)** | **r=16** | **2** | **-0.148** | **5/5** |

All 4 configurations LOSS. The r=2 LOSS magnitude (-0.148) is even
WORSE than r=1 (-0.081). Pattern: Muon r=1 ALWAYS drops capacity
(0.21 → 0.18), recovery in r=2 partial only.

**Mechanism update**: NS orthogonalization removes step-magnitude
information; this damage compounds across rounds because round-2's
chosen pool now contains the round-1-damaged completions. Multi-round
amplifies Muon's failure rather than rescues it.

### S4 — Multi-round Reverse-KL OPD LOSES (but partially rescued from SR catastrophe)

| OPD variant | substrate | rounds | mean ± σ | Δ vs SFT |
|---|---|---|---:|---:|
| Forward-KL OPD | HE 164 | 1 | 0.157 ± 0.070 | -0.088 |
| **Reverse-KL OPD** | HE 164 | 1 | 0.086 ± 0.045 | **-0.159** |
| **Reverse-KL OPD** | HE 164 | **2** | **0.260 ± 0.028** | **-0.145** |
| Hybrid OPD+SFT α=0.3 | HE 164 | 1 | 0.130 ± 0.088 | -0.114 |

Multi-round IS partial rescue: P16 S2 rev-KL OPD = 0.086 → P18 S4
MR-OPD = 0.260 (+0.17 absolute). But still LOSS vs MR-AdamW
(0.260 vs 0.404, Δ=-0.145).

Pattern: 4/5 seeds had r1 catastrophic drop (0.037-0.198), final-2
partial recovery to 0.23-0.29. Round-1 OPD KL pressure destroys
diversity; round-2 chosen pool from damaged model can't fully
rebuild.

**Risk #20 verdict**: falsified for both Muon AND OPD. The mechanisms
that caused Phase 14-16 retractions are NOT artifacts of single-round
protocol — they're rounds-independent. Phase 14-16 retractions
remain valid at multi-round too.

## Multi-round saturation curve — no plateau at r=4

| rounds | HE mean | σ | Δ vs prev | n |
|---:|---:|---:|---:|---:|
| 1 (P16 S1) | 0.230 | 0.031 | — | 5 |
| 2 (P17 S1) | 0.404 | 0.013 | +0.174 | 5 |
| **3 (P18 S2)** | **0.475** | **0.024** | **+0.070** | 5 |
| 4 (P18 S6) | 0.519 | n/a | +0.044 | 1 |

Per-round Δ shrinks (+0.17 → +0.07 → +0.04) but still positive at r=4.
Cumulative lift r0→r4 = +0.306, matching base pass@10 = 0.524.
**rounds=4 SFT effectively achieves what base needed k=10 inference
samples for**.

Diminishing returns suggest asymptote ~0.55-0.60. r=5 would test
this but at ~1.5× compute cost.

## Cross-substrate multi-round (S3 + S5) — generalizes

| substrate | r=1 mean | r=2 mean | Δ |
|---|---:|---:|---:|
| HumanEval (P17 S1) | 0.230 | 0.404 | +0.174 |
| **MBPP (P17 Sb + P18 S3 full 5)** | **0.355** | **0.457 ± 0.013** | **+0.102** |

Pass@k on MR-trained models (S5 + S7):

| substrate | base pass@10 | MR-SFT pass@10 | Δ |
|---|---:|---:|---:|
| HumanEval (Sa + S7, 2 seeds) | 0.524 | **0.595** | +0.071 |
| MBPP (S5, 2 seeds) | 0.480 | **0.560** | +0.080 |

**Both substrates: MR-SFT lifts pass@10 above base**. Single-round
SFT trades pass@k for pass@1; multi-round preserves it. Substrate-
independent.

## What this means for Phase 18+ practice

Settled (don't repeat at LoRA self-improve scale):
- **Muon at LoRA**: definitively retracted across substrate × rank
  × rounds (4 configurations)
- **OPD variants**: definitively retracted across forward/reverse-KL
  × hybrid × rounds (4 configurations)
- **Label smoothing as regularizer**: retracted P17 S2

Established positive recipe:
- **rounds=2 or 3 multi-round SFT** is the default. r=4 still adds
  +0.04 but cost grows.
- **pass@k inference** for deployment when verifier available.
- **samples=6 per round** is the substrate-default (P16 S1 σ=0.031).

## Phase 19 candidates

The two robust techniques (multi-round + pass@k) are at saturation
or near it. Where to push next:

1. **Combined recipe at MBPP** — multi-round + pass@k on MBPP gives
   ~0.56 pass@10. Cumulative wallclock + compute budget for
   end-to-end deployment of self-improve.
2. **rounds=5+ at HumanEval** — find the real plateau.
3. **Best-of-n training-time harvest at MR** — does k=10 chosen pool
   at round-2/3 break diminishing returns?
4. **Substrate scale-up (Qwen 1.5B-Coder)** — does the multi-round
   compounding hold at larger model scale? Phase 18 candidate
   deferred from Phase 17 closeout.
5. **RL with pass@k reward** — train directly against the inference
   objective. Phase 17 candidate, requires new infra.

## See also

- `docs/phase17-closeout.md` — Phase 17 closeout (this Phase 18
  picks up its candidate list)
- `docs/phase16-closeout.md` — Phase 16 closeout (samples=6 default)
- `scripts/phase18_s{1-7}/` — all stage scripts + per-seed JSONs
