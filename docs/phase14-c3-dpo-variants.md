# Phase 14 C3 — DPO variants for LoRA at Qwen substrate

Phase 11 S5 (1M K9) reported two notable DPO results, both flagged by
Phase 13 S1's variance audit as needing 5-seed cross-batch
replication:

- **Hybrid SFT+DPO α=0.3, β=0.1**: r1 eval 18/24 (75%, project record
  for K9 1M); collapsed in subsequent rounds; final ≤ SFT.
- **Round-0-only DPO**: 1 round faster to reach SFT baseline (eval
  11/24); final == SFT.

C3 re-tests both at the quiet Phase 14 substrate (σ_AdamW=0.011,
13× tighter than K9 1M). Symmetric companion to C2 (which retracted
Muon decisively).

## Setup

- **Model / problems**: same as Phase 14 S1 (Qwen2.5-Coder-0.5B + LoRA
  r=16, α=32, q_proj+v_proj; 25 HumanEval-style problems)
- **Optimizer**: AdamW (per C2 verdict)
- **Reference model**: PEFT `model.disable_adapter()` ctx → frozen
  base. At LoRA init delta_B=0 → policy ≡ ref, so DPO is informative
  only after LoRA has moved.
- **Pairs**: per-prompt pass × fail enumeration, capped at 4
  pairs/prompt
- **Variants**: hybrid (α=0.3, β=0.1) and round0 (pure DPO at r=0,
  SFT for r=1+)
- **Seeds**: 5 each
- **Hardware**: GPU 2 (hybrid) + GPU 3 (round0) ~45 min wallclock

## Decision gate

2σ ≈ **0.022 absolute final-pass-rate delta vs SFT** (Phase 14 S1's
σ=0.011).

- |Δ| > 0.022 → robust win/loss
- |Δ| ≤ 0.022 → within noise (no algorithmic signal)

## Result

5 seeds × {SFT (S1 baseline), Hybrid α=0.3 β=0.1, Round-0-only DPO} =
15 runs (10 new + 5 reused).

| arm | r0 | r1 | r2 | final-3 | σ_final | Δ vs SFT |
|---|---:|---:|---:|---:|---:|---:|
| SFT (baseline) | 0.547 ± .026 | 0.801 ± .016 | 0.846 ± .004 | **0.851 ± .011** | 0.011 | — |
| Hybrid α=0.3 | 0.547 ± .026 | 0.814 ± .052 | 0.820 ± .048 | **0.780 ± .117** | **0.117** | −0.071 |
| Round-0-only | 0.547 ± .026 | 0.764 ± .029 | 0.828 ± .033 | **0.838 ± .027** | 0.027 | −0.013 |

Per-seed final pass rate:

| seed | SFT | Hybrid | Round0 |
|---:|---:|---:|---:|
| 0 | 0.850 | **0.575** | 0.815 |
| 1 | 0.870 | 0.840 | 0.840 |
| 2 | 0.845 | 0.855 | 0.815 |
| 3 | 0.850 | 0.840 | 0.880 |
| 4 | 0.840 | 0.790 | 0.840 |

### Headline — hybrid catastrophically collapses on 1 seed in 5

Hybrid σ = **0.117**, 10.6× the SFT σ floor (0.011). One seed (0)
collapsed from r=1 peak 0.865 → r=2 0.735 → final 0.575 (back to
round-0 baseline). The other 4 seeds stayed in the 0.79-0.86 band.
This reproduces Phase 11 S5's K9 1M "r=1 spike + multi-round collapse"
signature — but K9 1M's noise floor (σ ≈ 0.142 within-batch) hid
*which seeds* collapsed; at quiet substrate, the failure mode is now
visible per-seed.

### Round-0-only — no acceleration, no catastrophe

Round0 σ = 0.027 (2.5× S1 floor), Δ_final = −0.013 (within noise).
Phase 11 S5 K9 claim of "1 round faster to SFT baseline" does **not**
replicate — round1 is actually slightly slower (0.764 vs SFT 0.801),
recovering to SFT-equivalent by final-3.

### Significance verdict

- |Δ_hybrid| = 0.071 vs 2σ_max = 0.234 → **WITHIN NOISE** at the
  inflated hybrid threshold, but the inflation itself is the finding:
  hybrid is unstable.
- |Δ_round0| = 0.013 vs 2σ_max = 0.053 → **WITHIN NOISE**, true
  SFT-equivalent.

### Pair count dynamics

DPO needs (chosen, rejected) pairs from per-prompt pass × fail
enumeration. Pair count drops sharply as model saturates:

| round | pairs |
|---|---:|
| r0 | 37 |
| r1 | 3-7 |
| r2 | 0-5 |
| r3 | 0-7 |

After r=0, hybrid's DPO term is operating on ≤7 pairs/round —
high-variance updates that occasionally derail training. Round-0-only
sidesteps this by dropping DPO once pairs go thin.

## Verdict — Phase 11 S5 K9 1M claims fail to replicate

Both Phase 11 S5 claims at K9 1M scale were noise:

1. **Hybrid α=0.3 r1 75% peak (project record)** — at Qwen substrate
   r1 mean is 0.814 ± 0.052, +0.013 vs SFT 0.801 — within 1σ_hybrid.
   Not a peak.
2. **Round-0-only 1 round faster to SFT baseline** — at Qwen
   substrate r1 is 0.764 vs SFT 0.801 — *slower* by −0.037, then
   catches up by final-3.

But C3 also surfaces a new failure mode that K9's noise hid:

3. **Hybrid σ blows up 10×** — 1/5 seeds collapses to round-0
   baseline. Catastrophic collapse is real, just intermittent.

### Decision impact

- **DPO is not added to Stage C default training**. AdamW + SFT
  remains the canonical recipe at Qwen substrate.
- **`nanogpt-rs::train_dpo` and curator preference-pair pipeline**
  stay in the codebase as Phase 11 prep — they work, they're
  unit-tested, but they don't help at LoRA-friendly substrate.
- **Phase 11 S5 retroactively retracted**: hybrid α=0.3's K9 record
  was noise; round-0-only's K9 acceleration was noise. Phase 13 S1's
  variance-audit prediction (single-run K9 1M claims need 5-seed
  cross-batch replication) confirmed for two more variants.
- **Risk register #17**: DPO on saturating substrates has insufficient
  rejected mass — pair count collapses after round 0, and what
  remains is high-variance and occasionally destructive.

### Why DPO doesn't help at quiet LoRA substrate

DPO needs failure mass to push against. Phase 14 substrate saturates
21/25 problems by round 1; rejected pile shrinks from 91 (r=0) to 27
(r=2) to 31 (r=3). High-quality SFT signal (the chosen-only pile) is
abundant and noiseless; DPO's contrastive update adds variance
without information.

The K9 1M failure mode where DPO mode-collapsed onto repeated tokens
(Phase 11 S3 risk #13) didn't appear here because LoRA's small-rank
constraint + frozen base limit how far the policy can drift. But the
pair-scarcity failure mode (this commit) is a substrate-shape issue
that LoRA doesn't solve.

## Reproducing

```bash
bash scripts/phase14_c3/run_hybrid.sh   # GPU 2, hybrid α=0.3 β=0.1, seeds 0-4
bash scripts/phase14_c3/run_round0.sh   # GPU 3, round0 β=0.1,        seeds 0-4
/tmp/p14_env/bin/python scripts/phase14_c3/analyze.py
```

## See also

- `docs/phase14-design.md` — Stage C plan
- `docs/phase14-s1-substrate.md` — substrate variance bound (SFT
  baseline)
- `docs/phase14-c2-muon-lora.md` — Muon retracted at LoRA scale
- `~/.claude/.../phase11_s5_hybrid_dpo.md` — Phase 11 S5 K9 1M result
- `scripts/phase14_c3/{self_improve.py, analyze.py}` — this commit
