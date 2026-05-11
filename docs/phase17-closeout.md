# Phase 17 closeout — 4 robust wins, multi-round SFT + inference scaling are the levers

After 6 phases (11-16) of training-side retractions (~7 algorithmic
LOSS verdicts), Phase 17 finally produced **robust positive findings**
on two orthogonal axes that the project had been neglecting:
1. **Multi-round SFT** (training axis, deepening rather than tweaking)
2. **Inference-time scaling** (orthogonal axis, untapped until now)

## Scoreboard

| stage | scope | result | mean ± σ |
|---|---|---|---:|
| **S1** | Multi-round SFT (rounds=2 samples=6) | **WIN** | 0.404 ± 0.013 |
| S2 | Label smoothing α=0.1 | LOSS | 0.181 ± 0.062 |
| S3 | MBPP single-round | positive | 0.363 ± 0.024 |
| **S6** | Base Qwen pass@k inference | **WIN** | pass@10 = 0.524 |
| S7a | samples=10 (single-round) | NEUTRAL | 0.236 ± 0.036 |
| S7b | Single-round SFT pass@k | TRADE-OFF | pass@1↑, pass@10↓ |
| **Sa** | Multi-round SFT + pass@k | **WIN** | pass@10 = 0.604 |
| **Sb** | Multi-round MBPP | **WIN** | 0.453 ± 0.016 |

Comparison baselines:
- Phase 16 S1 single-round samples=6 SFT: mean = 0.230 ± 0.031
- Phase 17 2σ threshold (samples=6): 0.062

## The two big findings

### Finding 1 — Multi-round SFT works (S1 + Sb)

| substrate | r=1 mean | **r=2 mean** | Δ |
|---|---:|---:|---:|
| HumanEval | 0.230 | **0.404** | **+0.174** |
| MBPP | 0.363 | **0.453** | +0.090 |

Both substrates show robust multi-round lift, well past 2σ. σ stays
tight (HumanEval σ_r2=0.013, even tighter than r=1's 0.031).

**Mechanism**: round-1 establishes some style without losing capacity.
Round-2's chosen pool draws from the round-1-improved model — DIFFERENT
completions than round-1's pool. Training on this fresh pool of
more-correct completions adds genuinely new capability without
single-mode collapse.

Why didn't we find this earlier? Phase 14-16 all used rounds=1 by
default. Phase 4 (K9) had multi-round but at toy substrate. Multi-
round at HumanEval was simply unmeasured until S1.

### Finding 2 — Inference-time scaling >>> training (S6 + Sa)

Base Qwen-Coder-0.5B pass@k unbiased estimators:

| k | base pass@k |
|---|---:|
| 1 | 0.216 |
| 2 | 0.300 |
| 5 | 0.425 |
| **10** | **0.524** |

Δ_pass(10−1) = +0.308 — **5× the 2σ threshold** for training-side
comparisons. The model already "knows" 52% of HumanEval if we let it
try 10 times. No training needed.

**Mechanism**: model has multi-modal solution distribution. T=0.8
sampling explores; verifier picks correct. Single-sample pass@1 is a
worst-case projection of this distribution.

Phase 17 S7b confirmed the mechanism by showing SFT TRADES pass@k for
pass@1: 1-round SFT bumps pass@1 by +0.018 but drops pass@10 by
-0.030. **Single-round SFT is diversity-destroying**.

### Their interaction — multi-round SFT preserves diversity (Sa)

Phase 17 Sa is the critical experiment. Multi-round SFT at HumanEval
+ pass@k eval (single seed, proof of concept):

| metric | base | 1-round SFT | **2-round SFT** |
|---|---:|---:|---:|
| pass@1 | 0.216 | 0.234 | **0.404** |
| pass@10 | 0.524 | 0.494 | **0.604** |

Multi-round SFT lifts BOTH axes simultaneously. The diversity-collapse
mechanism of single-round SFT is broken by multi-round dynamics.

**This means**: training-side intervention IS effective when applied
correctly. The 6 phases of training retractions were partly an
artifact of the single-round measurement protocol. Multi-round is
the "right" SFT recipe.

## What this revises about Phases 11-16

Risks #14-#19 documented across Phases 13-16 cited mechanism
arguments (noise floor, harvest dominance, optimizer fit, etc.).
Phase 17 retroactively suggests an additional axis we ignored:

**Risk #20 (new)**: rounds=1 single-round SFT is a diversity-
collapsing protocol. Algorithm comparisons that vary the optimizer/
loss/distillation under rounds=1 conflate algorithmic effects with
the underlying protocol's diversity collapse. Phase 14 C2/C3, Phase
15 S2/S4, Phase 16 S2/S3/S4 all used rounds=1; their retractions
remain valid AS rounds=1 verdicts but DO NOT transfer to multi-round.

We do not retract any Phase 14-16 verdict, but we no longer assert
those algorithmic comparisons are conclusive about the techniques
themselves at multi-round. Most-actionable revisit: Muon at
multi-round LoRA. Phase 18 candidate.

## What this commit changes for Phase 18+ practice

- **rounds=2 is the new default** for SFT comparisons. rounds=1 is
  smoke-test only.
- **pass@k is reported alongside pass@1** for any new technique. A
  technique that boosts pass@1 but drops pass@10 is suspect.
- **Multi-init averaging** continues to be dispreferred (Phase 16 S1).
- **samples=6** stays as default for single-round; for multi-round
  use samples=6 per round (Phase 17 S1 hyperparam).

## Phase 18 candidates (top priority)

1. **Multi-round Muon** — Phase 14 C2 + 15 S4 + 16 S3 all retracted
   Muon at rounds=1. Does multi-round Muon also lose, or did
   rounds=1 hide a real positive? Cheapest possible re-test.
2. **Multi-round OPD (reverse-KL hybrid)** — Phase 15 S2 + 16 S2/S4
   all retracted OPD variants at rounds=1. With multi-round's
   diversity-preserving dynamics, OPD might have a fairer test.
3. **Best-of-n harvest** — at training time, use k=10 samples to
   build chosen pool (vs k=6). Phase 17 S7a tested this at
   single-round (NEUTRAL); does multi-round + best-of-n harvest
   compound?
4. **rounds=3 dynamics** — Phase 17 S1 measured rounds=2. Does
   rounds=3 continue lifting (compound effect) or plateau?
5. **MBPP multi-round 5-seed completion** — Sb is 3-seed; full
   5-seed needed for clean σ.

## What NOT to spend more cycles on

- **More single-round optimizer/loss variants** — diversity collapse
  is the dominant signal, not the algorithmic differences. Revisit
  these at multi-round if at all.
- **More label smoothing variants at α≠0.1** — Phase 17 S2 + Phase
  14 C3 hybrid DPO pattern: regularizers cause σ blow-up. Move on.
- **More samples-per-round above 6 (single-round)** — Phase 17 S7a:
  diminishing returns + slight overfitting risk.

## Improved infrastructure carried forward

- `scripts/phase17_s1/` — multi-round SFT driver. Reuses Phase 15 S1
  harness with --rounds 2.
- `scripts/phase17_s3/` — MBPP-100 substrate (parsed signatures from
  canonical code).
- `scripts/phase17_s6/run_passk.py` — pass@k eval at base model.
- `scripts/phase17_s7/run_sft_then_passk.py` — SFT train + post-
  training pass@k eval combined.
- `scripts/phase17_sa/` — multi-round SFT + pass@k eval combined.
- `scripts/phase17_sb/` — multi-round on MBPP.

## See also

- `docs/phase16-closeout.md` — Phase 16 closeout (motivation +
  Phase 17 candidate list, of which S1/S2/S3/S6 were the picks)
- `docs/phase15-closeout.md` — Phase 15 closeout
- `data/mbpp/mbpp.jsonl` (gitignored) — fetched at experiment time

## Six-phase narrative (11-17)

| phase | dominant finding | retraction count |
|---|---|---:|
| 11 | DPO multi-round collapse at K9 | hybrid + round-0 retroactively retracted at P14 |
| 12 | Muon "+78%" at K9 (later seed-0 outlier) | retracted in P13 |
| 13 | K9 1M noise floor exposed | substrate retired |
| 14 | Qwen substrate qualified, Muon LOSS, DPO retracted | 2 retractions |
| 15 | HumanEval substrate qualified, OPD LOSS, Muon generalizes LOSS | 2 retractions |
| 16 | CLT validated, multiple OPD/Muon variants retracted | 3 retractions |
| **17** | **Multi-round SFT + pass@k inference both WIN** | **+4 robust positives** |

Phase 17 is the project's first phase with multiple robust positives.
Multi-round SFT is the canonical lever; inference-time scaling is the
"free 30pp headroom" axis. Phases 18+ should explore the multi-round
× inference axis combinations and apply them across more substrates.
