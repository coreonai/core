---
title: "Phase 22 — Qwen2.5-Coder-7B: migration + saturation + pass@k"
date: "2026-07-25"
---

# TL;DR

Migrated the Rust/Pekko self-improve stack from Qwen2.5-Coder-**0.5B** to
**7B** (single A100 40GB), then measured what actually helps a strong 7B
base on HumanEval:

- **Multi-round SFT: ~0 (saturated).** base 0.638 → r2 0.675 (baseline,
  +0.037 ± 0.085) / → 0.606 (gentler lr↓steps↓, −0.031 ± 0.043). Both flat
  within noise. **Not over-training** — the gentler recipe does slightly
  *worse*, not better. Contrast 0.5B: +0.25 over the same rounds.
- **pass@k (inference-time scaling): +0.347.** full-164 aggregate
  **pass@1 = 0.366 → per-prompt pass@10 = 0.713** (≈1.95×).
- **Verdict:** on a strong 7B base, the value comes from **inference-time
  scaling, not self-SFT**. The MR-SFT recipe's worth is a function of
  *headroom*, not the technique.

# Migration (code, all on origin/master)

| Piece | What | Commit |
|-------|------|--------|
| C1 | sharded-safetensors loader (`resolve_safetensors`, dual-mode file/dir) | `cbbfeb5` |
| C2 | `--model-dir` / `--model-id` override on the Phase-22 examples | `d954b98` |
| C3 | BF16 training path (`--train-bf16` / `--bf16`) | `fe63afd` |
| candle patch | vendored candle-core 0.10.2: matmul backward skips grads for frozen operands | `41ed3ce` |
| pipeline | CPU-side LoRA merge + free-before-load reload | `0ca067f` |
| trainer split | `--trainer-gpu` (inference + trainer on separate cards) | `d0b058c` |

Config/dimension adaptation is automatic (config.json → Qwen2Config;
tied/untied lm_head runtime-detected — 7B is untied, 0.5B tied).

Upstream: the candle fix is submitted as **huggingface/candle#3773**; drop
the `[patch.crates-io]` + `vendor/candle-core-0.10.2/` once it merges.

# Memory findings (measured on 7B, bf16)

- **Inference:** 15.1 GB (fits one 40GB card).
- **LoRA training multiplier:** candle 0.10.2 matmul backward computes+stores
  a gradient for BOTH operands unconditionally → for LoRA (~all weights
  frozen) it allocated a full weight-sized grad per frozen matmul → **~4×
  base** peak (0.5B 4.36GB, 1.5B 12.4GB; 7B predicted ~60GB → OOM). Patch
  (`if operand.track_op()`) cuts it to ~1.2× → **7B training 15.2GB**.
  Loss bit-identical; 179 tests pass.
- **Full pipeline residency:** inference model (15GB) + trainer (15GB) =
  ~30GB co-resident. Fixes: LoRA merge moved to CPU (was a 3rd 15GB GPU
  copy); `handle_reload` frees the old model before loading (was 3×15=45GB).
- **Training batch × seq:** batch=4 SFT on real HumanEval seqs (152K-vocab
  all-position logits + 28-layer MLP activations) peaks ~39GB **even with
  the trainer alone** on a card. Working config = **`--trainer-gpu` (2 GPUs)
  + `batch_size=2`** → trainer peak 27.8GB, model 15.1GB.

# Experiment: multi-round SFT saturation (4 seeds, rounds=3)

Recipe: Phase-17 (samples-per-prompt=6 → 984 attempts/round, max-new=200,
fresh-opt, non-cumulative harvest, top_k=0), two-GPU + batch=2, per-round
eval n=40 × passk=5 (directional; wide σ). base 0.638 ± 0.083 (same model).

**Baseline** (train-steps=100, lr=2e-4):

| Seed | base | r0 | r1 | r2 |
|------|------|----|----|----|
| 42 | 0.750 | 0.750 | 0.675 | 0.750 |
| 100 | 0.575 | 0.650 | 0.650 | 0.625 |
| 200 | 0.575 | 0.625 | 0.625 | 0.725 |
| 300 | 0.650 | 0.600 | 0.675 | 0.600 |
| **mean±σ** | 0.638±.083 | 0.656±.066 | 0.656±.024 | **0.675±.074** |

net Δ(base→r2) = **+0.037 ± 0.085** (< 0.5σ, not significant).

**Gentler** (train-steps=30, lr=5e-5):

| Seed | base | r0 | r1 | r2 |
|------|------|----|----|----|
| 42 | 0.750 | 0.725 | 0.725 | 0.700 |
| 100 | 0.575 | 0.550 | 0.550 | 0.550 |
| 200 | 0.575 | 0.550 | 0.575 | 0.600 |
| 300 | 0.650 | 0.675 | 0.550 | 0.575 |
| **mean±σ** | 0.638±.083 | 0.625±.089 | 0.600±.084 | **0.606±.066** |

net Δ(base→r2) = **−0.031 ± 0.043**. Gentler does *worse* → **over-training
ruled out; the base is saturated.**

# Experiment: pass@k (inference-time scaling)

`phase22_humaneval_baseline --model-id Qwen2.5-Coder-7B --n-problems 164
--passk 10 --sequential --aggregate` (8-GPU split by `--offset`):

| Metric | 7B | 0.5B (Phase 17 S6) |
|--------|----|--------------------|
| aggregate pass@1 | **0.366** (600/1640) | 0.216 |
| per-prompt pass@10 | **0.713** (117/164) | 0.524 |
| lift | **+0.347 (≈1.95×)** | +0.308 |

Headroom is in the hard tail: per-prompt pass@10 by problem index runs
~1.0/0.95 for idx 0–40 down to 0.43–0.59 for idx 100–164 — where any future
training gains would have to come from.

# Conclusions

1. The MR-SFT recipe that compounds a weak base (0.5B: 0.23→0.48) does
   **nothing** on a strong 7B base (HumanEval-saturated) — at either
   learning strength. This is **saturation, not over-training**.
2. **pass@k is the win on 7B** (+0.347), training-free — value comes from
   inference-time scaling.
3. Practical 7B recipe on 40GB: bf16, two-GPU (`--trainer-gpu`), batch=2,
   patched candle-core (or wait for #3773).

# Where next (not done)

- **Harder benchmark** with real 7B headroom (HumanEval/MBPP are near-ceiling
  for 7B) — to test whether self-improve helps a strong model *when there is
  room*, isolating headroom from model strength.
- **pass@k inside the Pekko actor stack** (Phase 21 Stage A wired it) on 7B.
- **RL / verifier-reward** on the hard tail (idx 100–164), the only headroom.
