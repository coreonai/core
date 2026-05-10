# Phase 16 S3 — Muon at LoRA r=64 also LOSES (rank-independent)

Phase 14 C2 + Phase 15 S4 retracted Muon for LoRA r=16 across both
saturated and headroom-rich substrates (Δ=−0.092 / −0.081). The
cited mechanism — "NS orthogonalization removes step-magnitude info
that small-rank LoRA needs" — explicitly invoked the *rank* as the
limiting factor. S3 tests whether r=64 (4× capacity) rescues Muon
or whether the mechanism is rank-independent.

## Setup

- HumanEval-164 + Qwen2.5-Coder-0.5B + LoRA, 1 round LoRA-FT, samples=3,
  train-steps=200
- LoRA `r=64, α=128` (preserves α/r=2 ratio from r=16 α=32)
- 5 seeds × Muon + matched 5 seeds × AdamW r=64 baseline (same
  hyperparams, different optimizer)
- Hardware: GPU 5 + 6 sequential, ~7h wallclock total

## Result

| arm | mean | σ |
|---|---:|---:|
| AdamW r=16 (Phase 15 S1) | 0.245 | 0.041 |
| Muon r=16 (Phase 15 S4) | 0.163 | 0.014 |
| **AdamW r=64** (this commit) | **0.233** | **0.097** |
| **Muon r=64** (this commit) | **0.162** | **0.018** |

### Δ within rank

| rank | Muon − AdamW |
|---|---:|
| r=16 | -0.081 |
| r=64 | -0.070 |

Both negative; magnitude similar.

### Δ across rank (does more capacity help?)

| optimizer | r=64 − r=16 |
|---|---:|
| AdamW | -0.012 |
| Muon | -0.001 |

Higher rank doesn't lift either optimizer at this substrate. AdamW
even gets slightly worse on average.

## Surprising side-finding — AdamW r=64 destabilizes

Per-seed AdamW r=64: [0.323, 0.220, **0.071**, 0.270, 0.278]. **Seed 2
catastrophically collapses to 0.071** (vs r=16 same seed = 0.224).
This pulls the mean down and inflates σ to 0.097, more than 2× r=16's
0.046.

| optimizer × rank | σ |
|---|---:|
| AdamW r=16 | 0.041 |
| **AdamW r=64** | **0.097** ← 2.4× wider |
| Muon r=16 | 0.014 |
| Muon r=64 | 0.018 (still tight) |

Higher LoRA rank gives AdamW more capacity and most seeds use it
beneficially (3/5 lifted vs r=16: +0.045, +0.075, ...), but capacity
becomes overfitting headroom for the unlucky seed. **r=64 doesn't
strictly dominate r=16 for AdamW** — the mean barely moves but
variance balloons.

This is a **rank-as-overfitting-knob** finding consistent with the
Phase 15 S1 mechanism (lift bimodality from overfitting). At r=16
overfitting hurts ~2/5 seeds mildly; at r=64 overfitting catastrophe
hurts ~1/5 seed severely.

Muon, by contrast, stays at σ≈0.018 across both ranks — its NS
orthogonalization regularizes against this overfitting failure mode
(stable but at a worse mean).

## Verdict — Muon mechanism is rank-independent

The Phase 14 C2 / Phase 15 S4 mechanism (NS orthogonalization
removes step-magnitude information that low-rank LoRA needs)
**generalizes to higher rank**. Three substrate × rank combinations
tested:

| substrate | rank | Δ Muon-AdamW |
|---|---|---:|
| Phase 14 (saturated) | r=16 | -0.092 |
| Phase 15 (headroom) | r=16 | -0.081 |
| Phase 15 (headroom) | r=64 | -0.070 |

All three negative; substrate × rank trends similar. Muon doesn't
work for LoRA self-improve at any combination tested.

The mechanism is therefore not "rank-r LoRA bottleneck" specifically
but rather "NS orthogonalization wrong inductive bias for LoRA at
this scale of training data." Updated mechanism hypothesis:

> Newton-Schulz produces orthogonalized step directions that
> equalize updates across LoRA singular values. With ~100-300
> chosen-pair training samples (Phase 15 substrate), the gradient
> already contains useful magnitude information about which
> directions matter. NS strips that, leaving the optimizer
> underspecified per direction. AdamW's per-parameter scaling is
> more appropriate at this data scale.

## Decision impact

- **Muon definitively retracted for Qwen-LoRA self-improve**.
  3 substrate × rank combinations LOSS, mechanism converging on
  data-scale reasons rather than rank specifically.
- **r=64 not a default for AdamW either**. The ~+0.023 mean lift
  on 4 of 5 seeds doesn't survive σ blowup. Phase 16+ stays at r=16.
- **Risk #19 (new)**: higher LoRA rank trades mean-lift on most
  seeds for catastrophic-collapse on minority seeds. Larger-capacity
  LoRA is overfitting-prone at this training data scale.

## Reproducing

```bash
bash scripts/phase16_s3/run_muon_r64_a.sh 5  # GPU 5, seeds 0/1/2
bash scripts/phase16_s3/run_muon_r64_b.sh 6  # GPU 6, seeds 3/4
bash scripts/phase16_s3/run_adam_r64_a.sh 5  # GPU 5, seeds 0/1/2 (after Muon)
bash scripts/phase16_s3/run_adam_r64_b.sh 6  # GPU 6, seeds 3/4
/tmp/p14_env/bin/python scripts/phase16_s3/analyze.py
```

## See also

- `docs/phase14-c2-muon-lora.md` — Phase 14 C2 (saturated, r=16)
- `docs/phase15-s4-muon-humaneval.md` — Phase 15 S4 (headroom, r=16)
- `docs/phase16-design.md` — Phase 16 plan including this re-test
- `nanogpt-rs/src/muon.rs` — Rust Muon implementation
- `scripts/phase14_c2/muon.py` — PyTorch Muon (reused by S3)
