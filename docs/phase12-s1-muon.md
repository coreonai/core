# Phase 12 S1 — Muon optimizer (DeepSeek V4 import)

DeepSeek V4 (2026-04-24) replaces AdamW with Muon (Newton-Schulz
orthogonalized SGD-momentum) for most modules at 1.6T scale. Phase
12 S1 ports Muon to nanogpt-rs and runs the K9 RustCode 4-round
falsifier test.

## Implementation

### `nanogpt-rs/src/muon.rs`

- `MuonConfig { lr, momentum, weight_decay, ns_steps, fallback_momentum }`
- `Muon` implements Candle's `Optimizer` trait — drop-in for
  `AdamW::new(...)`.
- 2-D weight matrices: gradient → momentum → Newton-Schulz orthogonalize
  → step.
- 1-D parameters (biases, LayerNorm scales): SGD-with-momentum
  fallback (no NS).
- Decoupled weight decay (AdamW-style).
- 5 NS iterations: 4 fast-stage `(3.4445, −4.7750, 2.0315)` + 1
  stabilize-stage `(2.0, −1.5, 0.5)`. Pre-normalizes by Frobenius
  norm so σ_max ≤ 1.

### `nanogpt-rs/src/train.rs::OptimizerKind` + `AnyOpt` enum

`TrainConfig.optimizer: OptimizerKind` (Adam | Muon, default Adam).
`train_from_full` dispatches via local `AnyOpt` enum (no trait
objects since `Optimizer::Config` is an associated type).

### CLI

- `train_kowiki_jepa --optimizer adam|muon`
- `self_improve_rust --optimizer adam|muon`

### Tests (+7 in `muon.rs` + 1 in `train.rs::cfg_persistence_tests`)

- `newton_schulz_identity_returns_near_identity`
- `newton_schulz_random_matrix_has_orthogonal_columns` (Q^T Q ≈ I)
- `newton_schulz_rejects_non_2d`
- `muon_step_decreases_quadratic_loss` (W ↓ ||W − target||² over 30 steps)
- `muon_set_learning_rate_round_trips`
- `muon_handles_1d_parameters_via_sgd_momentum_fallback`
- `train_from_full_runs_with_muon_optimizer` (CPU end-to-end smoke)

127 → **128** workspace tests (S1's standalone Muon tests; S2 OPD
adds 8 more in this same session — 136 total).

## Falsifier test — K9 RustCode 4 rounds

Both runs: 1500 pretrain + 4 rounds × 400 train_steps,
gen_n=24 / eval_n=24. Fresh pretrain seeds per optimizer. Two GPUs
parallel (~5 min/run).

| metric | AdamW | Muon | Δ |
|---|---:|---:|---:|
| r0 gen | 0/24 (0%) | **6/24 (25%)** | +25pp |
| r1 gen | 0/24 | **4/24 (16.7%)** | +16.7pp |
| r2 gen | **9/24 (37.5%)** | 6/24 (25%) | −12.5pp |
| r3 gen | 0/24 | 0/24 | 0 |
| **mean gen** | **9.4%** | **16.7%** | **+78% relative ★** |
| best eval | 5/24 | 2/24 | AdamW |
| final eval | 0/24 | 0/24 | tie |

### Reading

**Win**: Muon's stochastic generation is consistently above zero in
rounds 0/1/2 (6, 4, 6 of 24). AdamW posts zeros in three of four
rounds (only r2 spike). Mean gen-pass +78% relative.

**Loss**: Greedy eval is weaker. Muon's best eval is 2/24 vs
AdamW's 5/24. Final-round eval is 0/24 for both. AdamW retains a
slight edge on greedy decoding.

**Mechanism hypothesis**: Muon's gradient orthogonalization flattens
update magnitudes across directions — preserves diversity in the
output distribution but reduces argmax sharpness. At toy 1M scale,
this trade is roughly neutral for K9's slot-fill task.

### Caveat

Both runs underperform Phase 11 SFT baseline (final eval 11/24)
because they use **fresh pretrain seeds** (`p12s1_adam_seed`,
`p12s1_muon_seed`) rather than the cached `rust_seed.safetensors`.
The Muon vs AdamW comparison within this run is apples-to-apples;
the cross-comparison to Phase 11 isn't.

## Decision

**Adopt Muon as an optional axis, not the default.**

- **Phase 13+ NAS**: add `optimizer ∈ {Adam, Muon}` as the 13th axis
  of `GPTConfig`. Evolution can pick per-variant.
- **Default for training**: stay on AdamW. The greedy-eval edge
  matters more for our typical evaluation pipeline (eval is greedy
  by default in `self_improve_rust`).
- **Where Muon shines**: stochastic-generation use cases — agentic
  loops with temperature sampling, ensemble curation. Phase 11 S5's
  hybrid α=0.3 + Muon could be an interesting pairing (single-round
  eval spike + sustained gen diversity).

## Decision matrix vs Phase 12 design doc

| Phase 12 design gate | Result |
|---|---|
| Muon ≥ AdamW final eval | Tie (both 0/24) |
| Muon train loss faster convergence | Not measured this run — can extract from per-step loss curves |
| Muon stochastic gen >> AdamW | **Yes (+78% relative)** ★ |

## Risk #14 candidate

> Muon vs AdamW trade is **diversity ↔ sharpness**. Don't replace
> AdamW wholesale; deploy Muon for stochastic-output use cases
> (agentic, ensemble) and keep AdamW for greedy-eval use cases.

(Adding to `docs/phase7-design.md` as risk #14 in the next commit
if S2 OPD result confirms a pattern; otherwise S1 alone.)

## Reproducing

```bash
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p llm-actors --example self_improve_rust --features cuda --release

# AdamW control (GPU 0)
bash scripts/phase12_s1/run_adam.sh

# Muon (GPU 1)
bash scripts/phase12_s1/run_muon.sh

# Logs in scripts/phase12_s1/log_{adam,muon}.txt
```

## See also

- `docs/phase12-design.md` — overall Phase 12 plan + sequencing
- `nanogpt-rs/src/muon.rs` — implementation
- DeepSeek V4 technical report (Notion: workLLM Phase 10 S3 + 11 S1–S5 page)
