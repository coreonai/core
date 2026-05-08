# Phase 13 S1 — Variance bound on Muon vs AdamW (Phase 12 S1 framing overturned)

Phase 12 S1 reported a **+78% Muon gen-pass win over AdamW** on K9 — a
single 4-round comparison. Phase 13 S1's A2 step replicates that
comparison across **5 fresh pretrain seeds × {AdamW, Muon}** and
finds the win was a single-seed artifact.

## Setup

- 10 runs total: 5 seeds × {AdamW, Muon}
- Each run: 1500 pretrain steps + 4 self-improve rounds (gen_n=24,
  eval_n=24, round_train_steps=400)
- Two A100s in parallel — GPU 0 = AdamW × 5 seeds sequential, GPU
  1 = Muon × 5 seeds sequential
- Total wallclock ~25 min
- Same K9 3-challenge setup as Phase 12 S1 (rust_seed →
  fresh per-seed seed checkpoint)

## Result

### Per-seed mean gen-pass (sum-of-rounds gen / 96)

| seed | AdamW | Muon |
|--:|--:|--:|
| 0 | 0.083 | **0.240** ★ |
| 1 | 0.062 | 0.094 |
| 2 | 0.073 | 0.062 |
| 3 | 0.094 | 0.031 |
| 4 | 0.073 | 0.052 |

**AdamW: 0.077 ± 0.012** (tight)
**Muon:  0.096 ± 0.083** (very wide — driven by seed 0)

### Per-seed final eval/24

| seed | AdamW | Muon |
|--:|--:|--:|
| 0 | 2 | 0 |
| 1 | 5 | 0 |
| 2 | 3 | 0 |
| 3 | 8 | 0 |
| 4 | 10 | 0 |

**AdamW: 5.6 ± 3.4** (range 2-10, never 0)
**Muon:  0 ± 0** (all 5 seeds final eval = 0)

### Per-seed best eval/24 (max across rounds)

**AdamW: 9.8 ± 1.6** (range 8-11)
**Muon:  5.6 ± 0.9** (range 4-6)

## Verdicts

| Metric | Δ (Muon − AdamW) | z-score | Verdict |
|---|---:|---:|---|
| mean_gen_pass | +0.019 | +0.31 | **NOISE** within 1σ |
| final_eval/24 | **−5.60** | **−2.36** | **ROBUST AdamW win** |
| best_eval/24 | −4.20 | — | Consistent AdamW win |

## What Phase 12 S1 actually measured

Seed 0 happens to give Muon `mean_gen_pass = 0.240` (24% — a 3-4×
spike above its own per-seed mean). That single seed drove the
"+78% relative" headline. Across 5 seeds, Muon's mean (0.096) is
within 1σ of AdamW's mean (0.077). **Not a robust signal.**

Meanwhile the *negative* result on greedy eval **does** survive
variance:
- AdamW final eval consistently > 0 (5/5 seeds nonzero, mean 5.6)
- Muon final eval consistently = 0 (5/5 seeds)
- z = −2.36 → ROBUST

So Phase 12 S1's mixed framing ("+78% gen / slight greedy
regression") simplifies under variance to: **AdamW wins, Muon
loses, by ~half a model's worth of greedy eval performance**.

## Mechanism (revised)

The earlier "diversity ↔ sharpness trade" hypothesis was based on
the +78%-gen / −60%-eval data. Now we see Muon is *also* losing on
gen at the mean — the trade hypothesis doesn't hold.

A more parsimonious explanation: **Muon at this 1M scale just
underperforms AdamW**. Possible reasons:
- 5 NS iterations on small (100k-class) weight matrices are
  excessive — orthogonalization may amplify noise rather than smooth
  gradients
- DeepSeek's 1.6T scale: matrices are 10000×10000+; NS works well
  on dense, well-conditioned matrices. Our 1024×512 attention
  matrices are smaller and possibly less Newton-Schulz-friendly
- Hybrid AdamW+Muon (DeepSeek's actual pattern — Muon for 2-D,
  AdamW for 1-D) might still help, but our blanket Muon-for-2-D /
  SGD-mom-for-1-D is a different config

## Phase 12 S1 design-gate revisit

Phase 12 design doc decision gate was:
> Muon ≥ AdamW final eval → tie acceptable, NAS axis adoption

Phase 12 S1 single-run reported "tie (both 0/24)" → suggested
adoption. **Phase 13 S1 5-seed shows Muon final eval is** *robustly
less* **than AdamW (5.6 vs 0)** → adoption rejected.

**Decision: Muon is NOT added as a NAS axis. AdamW remains default.
The +78% gen-pass headline is retracted as a single-seed artifact.**

## Phase 11 SFT baseline reconciliation

Phase 11 SFT reported final eval 11/24. This Phase 13 S1 AdamW
mean is 5.6 ± 3.4 (max 10, never 11). Two factors:

1. **Phase 11 used cached `rust_seed.safetensors`** (one specific
   pretrain). Phase 13 S1 uses 5 fresh pretrains. The "11/24"
   was one good seed; Phase 13 S1 confirms the typical seed gives
   eval 5-10, not 11.
2. **K9 final-eval at this 1M scale has 2σ ≈ 7/24** — wider than
   most differences between optimizer choices. Comparing optimizers
   needs ≥ 5-10 seeds to detect anything below this threshold.

This reconciliation explains why **single-run optimizer / RL
comparisons throughout Phase 11/12 had high variance**. Phase 13 S1
formalizes this: K9 4-round 1M is **too noisy a measurement
substrate** for fine-grained algorithmic comparisons.

## Risk #14 — Single-run K9 optimizer/RL comparisons are unreliable

Adding to `docs/phase7-design.md`:

> K9 RustCode 4-round 1M-scale measurement has σ ≈ 3.4 / 24 on
> final eval and σ ≈ 0.08 on mean_gen_pass — a single run can
> easily generate a +78% / −60% relative delta that is pure
> sampling noise. **All algorithmic comparisons (optimizer, RL
> variant, distillation strategy) at K9 1M need ≥ 5 seeds before a
> claim can be promoted.** Phase 11 / Phase 12 S1 single-run
> claims are now retroactively flagged as 1σ noise unless backed
> by multi-seed measurement.

## Phase 13 next step (per design doc)

Design doc decision gate was:
- A2 noise → Stage B (200M)
- A2 robust → Stage A continues (DPO matrix variance, etc.)

Result: **A2 noise on gen, robust AdamW win on eval**. Decision:
proceed to Stage B (200M) — the question becomes whether 200M is
quieter substrate for these comparisons OR whether 1M-K9 noise is
inherent to the toy task, with a separate path C exploring real
benchmarks.

## What this commit does NOT change

- Muon module + tests stay in the codebase. They're correct, the
  optimizer works. We just don't claim it beats AdamW at K9 1M.
- OPD module from Phase 12 S2-loss stays. S3's OPD trainer
  measurement isn't affected by the Muon retraction.
- Phase 11 hybrid α=0.3 r1 18/24 — still claimed but **must be**
  retroactively considered single-seed (Phase 13 S1 A3 will
  re-measure with 5 seeds).

## Reproducing

```bash
# Two GPUs in parallel, ~25 min wallclock
bash scripts/phase13_s1/run_adam_seeds.sh
bash scripts/phase13_s1/run_muon_seeds.sh

# Analyze
python3 scripts/phase13_s1/analyze.py
```

## See also

- `docs/phase12-s1-muon.md` — original (now-retracted) +78% claim
- `docs/phase12-design.md` — Phase 12 sequencing (S2 OPD still
  active, just not building on S1's "Muon win")
- `docs/phase13-design.md` — Stage A → B sequencing
- `docs/phase7-design.md` — risk #14 added
