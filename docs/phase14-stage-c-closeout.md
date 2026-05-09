# Phase 14 Stage C — closeout

Stage C set out to (a) qualify a quieter substrate than K9 1M for
algorithmic comparisons, then (b) re-test K9-1M-noise-bound claims
that Phase 13 S1's variance audit had flagged. Three claims tested,
three retracted, plus one substrate-shape failure mode surfaced that
K9 1M's noise had hidden. C4 (OPD) deferred — substrate too saturated
to test OPD's multi-teacher claim meaningfully.

## Scoreboard

| stage | scope | result | commit |
|---|---|---|---|
| C1 | substrate variance | ROBUST σ=0.011 (13× tighter than K9 1M) | bb3a472 |
| C2 | Muon for LoRA | **LOSS Δ=−0.092 (4× threshold)** | 284c780 |
| C3 | DPO variants (hybrid α=0.3, round-0-only) | **retracted, hybrid σ blowup 10×** | 32d4c89 |
| C4 | OPD vs SFT | **deferred** — opd.py ported; full measurement → Phase 15 | this commit |

## What we learned

### Falsifier discipline + quiet substrate compounds

Phase 13 S1's variance audit retracted Phase 12 S1's "+78% Muon" as
a seed-0 K9-noise outlier; Phase 14 C2 confirmed at higher SNR that
Muon is *actively worse* than AdamW for LoRA. Phase 14 C3 retracted
two more Phase 11 S5 K9-noise claims (hybrid 75% peak; round-0-only
acceleration). Without the Phase 14 substrate, those three would
still be pending open questions.

### K9 1M was hiding *which* seeds collapsed

C3 hybrid α=0.3 reproduced the Phase 11 S5 "r=1 spike + multi-round
collapse" signature, but at K9's σ=0.142 you couldn't tell *which*
seeds collapsed. At Qwen substrate the failure mode is per-seed
visible: 1/5 seeds (seed 0) goes from r=1 peak 0.865 to final 0.575.
Variance blowup (10× SFT σ) is the real finding, not the mean.

### Saturating single-task substrate has its own failure modes

C2's mechanism: NS orthogonalization removes step-magnitude
information that small-rank LoRA needs to lock in deterministic
completions. Muon spread → wrong inductive bias for LoRA.

C3's mechanism: 21/25 problems saturate by r=1. Pair count drops
37→3-7→0-7. DPO's contrastive update on ≤7 pairs/round becomes
high-variance noise; hybrid keeps it active and occasionally derails.

These are substrate-shape failures (saturating + thin-rejected-pile),
not optimizer-shape, so changing optimizer (Muon) or contrastive
target (DPO) doesn't help. Risks #16 (optimizer non-monotonic
LoRA-vs-full) + #17 (DPO needs failure mass) added.

### OPD needs a different substrate

C4 was meant to port DeepSeek V4's On-Policy Distillation and test
"unified student trained on own rollouts vs frozen specialist
teachers" against the C1+C2+C3 backdrop. At Qwen + 25-problem
substrate two structural problems block this:

1. **No specialist axis.** All 25 problems are single-line Python
   completions; subdividing into "arithmetic / predicate / string"
   gives buckets too small for separate teachers (5-10 problems
   each, 84% saturation under SFT).
2. **No teacher headroom.** Qwen 0.5B + LoRA already saturates 21/25
   under SFT. OPD's lift comes from teacher knowing things student
   doesn't — saturation eliminates that gap.

Falling back to "OPD as KL-anchor regularizer" (single teacher =
frozen base Qwen) reduces to KL-regularized SFT, which is
mechanistically similar to DPO with reference. C3 already showed
DPO doesn't help here; the regularizer test is almost guaranteed
null.

C4 is therefore deferred to Phase 15 with a harder, multi-domain
benchmark. Phase 12 S2's "trainer deferred" debt is partially
addressed by `scripts/phase14_c4/opd.py` (PyTorch port + 6 self-
tests, mirror of `nanogpt-rs/src/opd.rs`).

## What's in this commit

- `scripts/phase14_c4/opd.py` — PyTorch OPD loss with forward /
  reverse KL, multi-teacher weighted sum, label-mask, full-vocab.
  6 self-tests pass: KL(p||p)=0 (forward + reverse), disagreement
  → large positive, weighted-sum decomposition, label-mask respect,
  temperature-softens, gradient flows to student only.
- `docs/phase14-stage-c-closeout.md` — this doc.

## Phase 15 — proposed direction

The Phase 14 substrate accomplished its goal but ran into its
saturation ceiling. Phase 15 should:

1. **Harder benchmark**. MBPP / HumanEval-full (164 problems) or a
   multi-domain mix (Python + Rust + JS one-liners). Aim: SFT
   baseline ≈ 50-70% (movable headroom), σ ≤ 0.03.
2. **Multi-teacher OPD test**. Train k=2-3 specialists on disjoint
   problem subsets; OPD a unified student vs SFT-on-union baseline.
   This is the real OPD test C4 couldn't be.
3. **Cross-axis variance audit**. Phase 14 S1 acknowledged σ=0.011
   is a *lower bound* on substrate noise (LoRA-init RNG and harvest
   RNG are entangled per seed). Add temperature/checkpoint-axis
   variance measurement before any new algorithmic claim.

## See also

- `docs/phase14-design.md` — original Stage C plan
- `docs/phase14-s1-substrate.md` — substrate qualification (C1)
- `docs/phase14-c2-muon-lora.md` — Muon retraction (C2)
- `docs/phase14-c3-dpo-variants.md` — DPO retraction (C3)
- `nanogpt-rs/src/opd.rs` — Rust OPD loss (Phase 12 S2)
- `scripts/phase14_c4/opd.py` — PyTorch OPD loss (this commit)
