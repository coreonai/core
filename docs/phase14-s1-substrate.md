# Phase 14 S1 — Qwen substrate variance bound (ROBUST positive)

Phase 13 closed with K9 1M retired as algorithmic-comparison
substrate (cross-batch σ ≈ 0.292 / 24 swallowed most algorithmic
deltas). Phase 14 S1 measures variance at the **Phase 9 substrate**
(Qwen2.5-Coder-0.5B + LoRA + 25 HumanEval-style problems) to verify
it is quieter before investing in C2-C4 algorithmic comparisons.

## Setup

- **Model**: Qwen2.5-Coder-0.5B + LoRA (r=16, α=32, q_proj+v_proj)
- **Problem set**: 25 HumanEval-style single-line completions
  (11 carried over from Phase 9 S5 + 14 new)
- **Self-improve protocol**: 3 rounds × (8 samples / problem,
  verifier pass → LoRA-FT 60 steps with AdamW)
- **Seeds**: 5 (controls torch global RNG for LoRA init AND
  seed_base offset for harvest sampling)
- **Hardware**: 2 A100s parallel (GPU 0 = seeds 0/1/2 sequential,
  GPU 1 = seeds 3/4 sequential), ~45 min wallclock

## Result — variance bound

Per-round mean ± σ pass rate (over 200 trials = 25 problems × 8 samples):

| round | mean | σ | per-seed |
|---|---:|---:|:---|
| 0 (pre-train) | 0.547 | 0.026 | [0.575, 0.565, 0.535, 0.550, 0.510] |
| 1 | 0.801 | 0.016 | [0.820, 0.790, 0.810, 0.805, 0.780] |
| 2 | 0.846 | 0.004 | [0.850, 0.845, 0.845, 0.840, 0.850] |
| **final-3** | **0.851** | **0.011** | [0.850, 0.870, 0.845, 0.850, 0.840] |

## Substrate verdict — ROBUST

| | Final pass rate σ |
|---|---:|
| **Phase 14 Qwen substrate** | **0.011** |
| K9 1M within-batch | 0.142 (~13× wider) |
| K9 1M cross-batch | 0.292 (~27× wider) |

Phase 14 substrate is **substantially quieter** than K9 1M. σ ≪
the noise floor that swallowed Phase 11/12 algorithmic claims at
K9. Substrate is **qualified** for C2/C3/C4.

## Per-challenge composition (final round)

| bucket | count | examples |
|---|--:|---|
| Saturated (100% pass, σ ≤ 0.06) | **21** | max_of_two, is_even, square, fizz_string, list_max … |
| Headroom (mid pass with σ) | 1 | equals_5 (0.30 ± 0.34) |
| Cold-start (0% across all seeds) | 3 | equals_14_via_doubling, len_5_string, ten_minus_to_3 |

The 3 cold-start problems are exactly the same trio Phase 9 S5
identified — they require domain-specific guessing (e.g. `2 * (7)`
to land on 14) that Qwen-Coder's prior doesn't include. Curriculum
or hand-injection (Phase 9 S5 risk #11) needed.

## Caveats

### Saturation ceiling

21/25 saturated means most algorithmic comparisons (C2 Muon, C3
DPO, C4 OPD) would barely move overall pass rate — only 4
problems offer headroom (1 mid + 3 cold-start). For Phase 14
algorithmic comparisons, the right metric is:

1. **Movement on the 4 non-saturated problems** specifically
   (focused subset eval), or
2. **A harder problem set** drawn from real HumanEval / MBPP

C2-C4 measurements should also report focused-subset σ in addition
to overall σ.

### LoRA init randomness vs harvest randomness

Each seed controls both:
- `torch.manual_seed(args.seed)` — LoRA delta_A / delta_B init
- `seed_base = args.seed * 1_000_000` — generation sampling

These are entangled. A "true" variance bound would also vary
- pretrain checkpoint (we use the same Qwen0.5B base for all 5)
- generation temperature / sampling RNG independently

The σ = 0.011 measurement is a *useful lower bound* on
Phase-14-substrate noise, not the full noise picture. Cross-axis
variance (different model checkpoints, different temperature
schedules) would be larger.

## Decision — proceed to Phase 14 C2

C1 verdict: **σ = 0.011 ≪ K9 σ. Substrate qualified.**

C2 plan: Muon for LoRA adapters at Qwen scale.
- Implement Muon orthogonalization for LoRA A/B matrices in Python
  (PyTorch). Rust nanogpt-rs::muon won't directly apply (different
  framework).
- 5 seeds × {AdamW (this S1 baseline), Muon} = 10 runs
- Compare: `p_muon - p_adam > 2 max(σ_adam, σ_muon)` for "robust win"
- σ_adam = 0.011 → significance threshold ≈ 0.022

If Muon delta on Qwen LoRA is ≥ 0.022 → genuine win at quiet
substrate (vindicates Phase 12 S1 partially, retroactively).
Below threshold → Muon doesn't help even at LoRA-friendly scale.

## What this commit changes

- **Algorithmic claims at Phase 9 substrate are credible** down
  to ~0.022 absolute pass-rate delta — far better than K9 1M's
  ~0.30 cross-batch threshold.
- **K9-noise-floor-bound retracted claims** (Phase 11 hybrid α=0.3
  r1=18/24, Phase 12 Muon +78%) can be re-tested at this
  substrate. C2 (Muon) and C3 (DPO variants) are the natural
  re-tests.
- **Substrate selection rule** for the project: when measuring
  algorithmic deltas, default to Qwen + 25-problem set; K9 1M is
  smoke-test only.

## Reproducing

```bash
# Setup (one-time)
python3 -m venv /tmp/p14_env
/tmp/p14_env/bin/pip install torch transformers peft accelerate

# Run 5 seeds
bash scripts/phase14_s1/run_seeds_a.sh   # GPU 0, seeds 0/1/2
bash scripts/phase14_s1/run_seeds_b.sh   # GPU 1, seeds 3/4

# Analyze
/tmp/p14_env/bin/python scripts/phase14_s1/analyze.py
```

## See also

- `docs/phase14-design.md` — Stage C 4-step plan
- `docs/phase13-s3-isolate-budget.md` — K9 retirement rationale
- `scripts/phase9_s5/` — predecessor self-improve script
- `scripts/phase14_s1/{problems.py, self_improve.py, analyze.py}` —
  this commit
