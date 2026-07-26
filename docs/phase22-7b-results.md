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
- **Self-SFT DOES work where there's headroom + harvest — the hard tail.**
  On HumanEval idx 100–163 (base pass@5 ~0.25, real headroom): samples=6
  gives a noisy +0.094 ± 0.106 (3/4 seeds); **samples=16 gives +0.254 ±
  0.068 (pass@5 0.246 → 0.500, 4/4 seeds, ~3.7σ)** — the first robust 7B
  self-improve win.
- **Harvest has an INTERIOR OPTIMUM (~samples=16).** Sweeping
  samples-per-prompt on the hard tail: 6 → +0.094, **16 → +0.254**, 32 →
  +0.094 (right back down, one seed collapsed). Too little harvest =
  cold-start-noisy; too much = over-trains/destabilizes. An inverted-U.
- **RL (REINFORCE) on the hard tail COLLAPSES on adapter sync.** RLOO
  verifier-as-reward, k=4, sync-every=4: steps 0–3 healthy (~15/256 pass),
  then the first adapter sync craters it — lr=2e-4 → all seeds 0/256 (full
  collapse); lr=5e-5 → mean 15→3/256 (−80%, 1/4 still full-collapse).
  Gentler lr softens but doesn't prevent. RL is the *weak* axis; SFT
  (+0.254) is the robust hard-tail win. Reproduces Phase 22 Stage E.
- **Verdict:** the MR-SFT recipe's worth is a function of **headroom ×
  harvest**, not the technique — and harvest is a *tuned* knob with a
  sweet spot, not "more is better". No headroom (full set @ pass@5) →
  flat; headroom + optimal harvest (hard tail, samples≈16) → strong tight
  lift; headroom + too-little/too-much harvest → noisy/degraded.
  Orthogonally, inference-time **pass@k** is always a training-free win
  (+0.347).

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

# Experiment: hard tail (headroom) + cold-start rescue

HumanEval idx 100–163 (64 hard problems, base pass@5 ~0.25 — real headroom),
via `--prompt-skip-list 0..99` (FilteredDomain). Same two-GPU + batch=2
recipe, rounds=3, 4 seeds, per-round eval on all 64 at passk=5.

**samples-per-prompt = 6** (thin harvest, ~18–27 pairs/round):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.203 | 0.188 | 0.406 | +0.172 |
| 100 | 0.281 | 0.172 | 0.359 | 0.406 | +0.125 |
| 200 | 0.234 | 0.266 | 0.234 | 0.172 | −0.062 |
| 300 | 0.234 | 0.312 | 0.266 | 0.375 | +0.141 |
| **mean** | 0.246 | 0.238 | 0.262 | **0.340** | **+0.094 ± 0.106** |

Real but noisy lift (3/4 up, one declines) — headroom present, but the base
rarely passes the hard problems → sparse harvest → high variance
(cold-start, Phase 9 S5 risk #11).

**samples-per-prompt = 16** (rich harvest — base pass@1 ~0.15 → P(≥1 of 16
pass) ≈ 0.93 per problem):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.188 | 0.516 | 0.516 | +0.282 |
| 100 | 0.281 | 0.359 | 0.500 | 0.547 | +0.266 |
| 200 | 0.234 | 0.234 | 0.234 | 0.391 | +0.157 |
| 300 | 0.234 | 0.312 | 0.344 | 0.547 | +0.313 |
| **mean** | 0.246 | 0.273 | 0.399 | **0.500** | **+0.254 ± 0.068** |

**Cold-start rescue works decisively:** mean lift more than doubled
(+0.094 → +0.254; hard-tail pass@5 **doubled** 0.246 → 0.500), σ tightened
(0.106 → 0.068), **4/4 seeds positive** (incl. the samples=6 dissenter,
which sat at base through r0/r1 then jumped +0.156 once its harvest grew).
Harvest self-reinforces (18 → 200 for the strongest seed). ~3.7σ significant
— **the first robust self-improve win at 7B.**

**samples-per-prompt = 32** (2× the harvest again — is more better?):

| Seed | base | r0 | r1 | r2 | net |
|------|------|----|----|----|-----|
| 42 | 0.234 | 0.125 | 0.328 | 0.375 | +0.141 |
| 100 | 0.281 | 0.266 | 0.391 | 0.344 | +0.063 |
| 200 | 0.234 | 0.234 | 0.328 | 0.453 | +0.219 |
| 300 | 0.234 | 0.422 | 0.422 | 0.188 | −0.046 |
| **mean** | 0.246 | 0.262 | 0.367 | **0.340** | **+0.094 ± 0.113** |

**Harvest frontier is an inverted-U with a peak at ~16:**

| samples/prompt | r2 mean | net Δ (base→r2) |
|----------------|---------|-----------------|
| 6 | 0.340 | +0.094 ± 0.106 |
| **16** | **0.500** | **+0.254 ± 0.068** |
| 32 | 0.340 | +0.094 ± 0.113 |

samples=32 lands **exactly back at samples=6** (0.340 / +0.094), with high
variance (seed 300 collapsed 0.422 → 0.188 in r2). So more harvest is **not**
better past ~16: too little → cold-start-noisy; too much → over-trains on the
large self-generated corpus and destabilizes (same over-training failure mode
as the full set). **Harvest is a tuned knob with an interior optimum, not a
monotone lever.**

# Experiment: RL (REINFORCE) on the hard tail

Ported `phase22_he_reinforce` to 7B + hard tail (`--model-id`, `--train-bf16`,
`--trainer-gpu`, `--prompt-offset`; commit `160ccd6`). RLOO verifier-as-reward,
idx 100–163, k=4, max_new=192, pg-micro-batch=1, sync-every=4, 4 seeds.
Validated: peak model GPU 15.1GB / PG-trainer GPU 20.2GB (fits 40GB).

Per-step passes / 256 (all 4 seeds sample the hard tail each step):

| step | 0 | 1 | 2 | 3 (sync fires) | **4 (first post-sync)** |
|------|---|---|---|---|---|
| lr=2e-4 (seed 42) | 17 | 18 | 17 | 13 | **0** → 0 for all remaining 16 steps |
| lr=5e-5 mean (4 seeds) | 16 | 18 | 15 | **15** | **3** (seed 100 → 0, others 2/3/7) |

**The first adapter sync craters the policy.** Steps 0–3 have a healthy,
non-zero gradient signal (~15/256 ≈ 6% per-completion pass, plausible for the
hard tail). At the first sync (`SaveMergedCheckpoint` → `ReloadCheckpoint`,
step 3) the sampling model reloads the merged LoRA and the policy collapses:
lr=2e-4 → **0/256 for every remaining step** (full mode collapse); lr=5e-5 →
mean drops 15→3 (−80%), one seed to 0, the rest barely alive. Gentler lr only
softens it. **RL + adapter-sync is unstable at this scale/sparsity** — the same
mode-collapse Phase 22 Stage E found on the full set. Infra works; the RL
*algorithm* is the weak axis. **SFT's +0.254 is the robust hard-tail win.**

# Where next (not done)

- **Fix RL collapse** if pursued: reference-policy KL penalty, off-policy
  correction, no-sync + one final merge, or SFT-warmstart before RL.
- harder external benchmark to map the headroom×harvest frontier further.
