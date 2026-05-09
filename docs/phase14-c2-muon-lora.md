# Phase 14 C2 — Muon vs AdamW for LoRA at Qwen substrate

Phase 12 S1 reported "+78% Muon gen" on K9 1M, retracted by Phase 13 S1
as a seed-0 outlier (cross-seed σ swallowed the effect). Phase 14 C2
re-tests that retraction at the **Qwen + 25-problem substrate**
qualified by Phase 14 S1 (σ = 0.011 — 13× tighter than K9 1M
within-batch).

## Setup

- **Model**: Qwen2.5-Coder-0.5B + LoRA (r=16, α=32, q_proj+v_proj)
- **Problem set**: 25 HumanEval-style single-line completions (Phase 9
  S5 carry-over + Phase 14 S1 additions)
- **Self-improve protocol**: 3 rounds × (8 samples / problem,
  verifier pass → LoRA-FT 60 steps)
- **Optimizer arms**: AdamW (Phase 14 S1 baseline, default `lr=2e-4`)
  vs Muon (`lr=2e-4`, momentum=0.95, weight_decay=0.01, ns_steps=5)
- **Seeds**: 5 each (controls torch global RNG + seed_base offset)
- **Hardware**: GPU 2 (seeds 0/1/2 sequential) + GPU 3 (seeds 3/4
  sequential), ~45 min wallclock

## Muon implementation

`scripts/phase14_c2/muon.py` mirrors `nanogpt-rs/src/muon.rs`:

- 5 Newton-Schulz iterations (4 fast + 1 stabilize)
- 2-D parameters: gradient → momentum → NS orthogonalize → step
- 1-D parameters: gradient → momentum → step (SGD-momentum fallback)
- Decoupled weight decay (AdamW-style)
- Dispatch fp32 NS computation when LoRA adapters are fp16

LoRA adapters (`q_proj`, `v_proj` delta_A and delta_B) are exactly the
small-dense 2-D matrices Muon's NS targets — the most ergonomic
substrate for testing the algorithm.

## Decision gate

Significance threshold (using Phase 14 S1 baseline σ as floor):

> 2σ ≈ **0.022 absolute final-pass-rate delta**

If `mean(Muon) − mean(AdamW) > +0.022` → **robust win**, partially
vindicates Phase 12 S1's retracted claim at LoRA scale.
If `< +0.022` → Muon doesn't help even at the LoRA-friendly substrate.

## Result — robust LOSS for Muon

Per-round mean ± σ pass rate (5 seeds, 200 trials per round = 25
problems × 8 samples):

| round | AdamW | Muon | Δ (Muon − AdamW) |
|---|---:|---:|---:|
| 0 (pre-train) | 0.547 ± 0.026 | 0.547 ± 0.026 | 0.000 (shared init) |
| 1 | 0.801 ± 0.016 | 0.636 ± 0.020 | **−0.165** |
| 2 | 0.846 ± 0.004 | 0.704 ± 0.026 | **−0.142** |
| **final-3** | **0.851 ± 0.011** | **0.759 ± 0.004** | **−0.092** |

Δ_final = **−0.092**, threshold = 2 max(σ) = 0.023. Muon trails AdamW
by **4× the significance threshold** — no ambiguity.

σ_Muon (0.004) is even *tighter* than σ_AdamW (0.011) — Muon's loss is
not noise, it's a stable trajectory below AdamW.

### Per-challenge focused-subset

The 4 non-saturated problems show no Muon win:

| problem | AdamW | Muon | Δ |
|---|---:|---:|---:|
| `equals_5` | 0.300 ± 0.338 | 0.125 ± 0.088 | **−0.175** |
| `equals_14_via_doubling` | 0.000 | 0.000 | 0.000 |
| `len_5_string` | 0.000 | 0.000 | 0.000 |
| `ten_minus_to_3` | 0.000 | 0.025 ± 0.056 | +0.025 |

Cold-start trio still cold-start under Muon.

### Muon breaks 6 saturated problems

| problem | AdamW | Muon |
|---|---:|---:|
| `two_plus_to_5` | 1.000 | **0.500** |
| `two_pow_to_8` | 1.000 | **0.550** |
| `count_chars` | 1.000 | 0.900 |
| `abs_value` | 1.000 | **0.275** |
| `list_length` | 1.000 | 0.775 |
| `fizz_string` | 1.000 | 0.925 |

Muon's NS orthogonalization is *removing capacity* on easy problems
that AdamW saturates. The orthogonal-step update direction throws away
the magnitude information that LoRA's small-rank updates need to lock
in deterministic completions.

## Per-challenge focused-subset

S1 saturated 21/25 problems → most algorithmic deltas live in the 4
non-saturated:

- `equals_5` (mid: AdamW 0.30 ± 0.34)
- `equals_14_via_doubling` (cold-start, AdamW 0.0)
- `len_5_string` (cold-start, AdamW 0.0)
- `ten_minus_to_3` (cold-start, AdamW 0.0)

If Muon cracks any cold-start, that's a per-problem signal even if
overall Δ is below threshold.

## Verdict — Muon retracted at LoRA scale too

Phase 12 S1's "+78% Muon gen" was a K9-noise-floor seed-0 outlier
(retracted by Phase 13 S1). Phase 14 C2 now confirms at the quiet
Qwen substrate that **Muon actively hurts LoRA training**, not just
fails to help.

Three independent signals:

1. **Final pass rate**: −0.092, ~4× the noise threshold.
2. **Round-1 trajectory**: AdamW 0.801 vs Muon 0.636 — Muon is
   slower from the very first LoRA-FT step, not just plateauing
   lower.
3. **Saturated-problem regression**: Muon de-saturates 6 problems
   that AdamW handles cleanly. NS orthogonalization removes the
   step-magnitude information small-rank LoRA updates need.

### Why Muon hurts LoRA specifically

LoRA delta_A / delta_B are rank-r matrices (r=16) projecting through
a high-dim shared vocab. Their gradient already lives in a tightly
coupled manifold. NS orthogonalization rewrites the step direction
to be uniformly-distributed across singular directions — but the
*right* update for these deterministic completions concentrates
mass on a few directions. Muon's "spread" is the wrong inductive
bias for LoRA.

For full-finetune at much larger scale (e.g. DeepSeek V4's reported
use), the spread may average out usefully across many parameters.
At LoRA scale it's a destructive averaging.

### What this commit changes

- **Muon is not added as a default LoRA optimizer.** AdamW remains
  the canonical Phase 14 substrate optimizer.
- **Phase 12 S1's claim is doubly retracted**: K9 noise (Phase 13
  S1) + actively-worse-when-quiet (Phase 14 C2).
- **Stage C C3/C4 will use AdamW.** No need to re-run S1 baseline
  with Muon for fairness — Muon is decisively worse.
- **Risk register addition**: "Optimizer transfer from full-finetune
  → LoRA is non-monotonic." — DeepSeek V4 Muon results don't carry
  to LoRA without adaptation.

The Rust port `nanogpt-rs/src/muon.rs` stays in the codebase as an
NAS axis option (Phase 12 S1 added it as 13th axis), but Phase 14
C2 closes the question for LoRA: don't enable it.

## Reproducing

```bash
bash scripts/phase14_c2/run_muon_a.sh   # GPU 2, seeds 0/1/2
bash scripts/phase14_c2/run_muon_b.sh   # GPU 3, seeds 3/4
/tmp/p14_env/bin/python scripts/phase14_c2/analyze.py
```

## See also

- `docs/phase14-design.md` — Stage C plan
- `docs/phase14-s1-substrate.md` — substrate variance bound
- `scripts/phase14_c2/{muon.py, self_improve.py, analyze.py}` — this commit
- `nanogpt-rs/src/muon.rs` — Rust port (Phase 12 S1, retracted)
- `~/.claude/.../phase13_s1_variance.md` — original retraction
