# Phase 20 S3 — Deployment compute/timing budget

Production-ready recipes from Phase 17-19 cumulative findings. Pulls
numbers from `phase17-closeout.md` + `phase18-closeout.md` +
`phase19-closeout.md`. **No new measurement** — this is the recipe
distillation step.

## TL;DR — recipe selector

| budget | recipe | pass-rate | notes |
|---|---|---:|---|
| **cheap** | base + pass@5 | ~0.45 | no training; samples=5 at inference |
| **balanced** | r=2 SFT + pass@5 | ~0.55 | 1 LoRA train + 5 samples |
| **best** | r=3 SFT + pass@5 | **0.567** | 2 LoRA trains + 5 samples |
| **research-max** | r=5 SFT + pass@10 | ~0.65 est | 4 LoRA trains + 10 samples |

Pareto front explanation below.

## Substrate

- **Model**: Qwen2.5-Coder-0.5B (HF)
- **Adapter**: LoRA r=16, α=32, `q_proj` + `v_proj`, dropout=0
- **Training**: AdamW, lr=2e-4, batch_size=4, train_steps=200/round
- **Harvest**: samples=6/prompt, temperature=0.8, top_p=0.95, max_new=200
- **Eval substrate**: HumanEval-164 (primary), MBPP-100 (cross-substrate)

## Training cost per recipe (HumanEval)

Wall-clock measured on 1 × A100-40GB. Each harvest round = forward
pass over 164 prompts × 6 samples + cargo-style verify ≈ **80 min**.
LoRA fine-tune step = ~30s.

| recipe | harvest rounds | LoRA trains | GPU-hours | wallclock (1 GPU) |
|---|---:|---:|---:|---:|
| base only | 1 (eval) | 0 | 1.3 | 80 min |
| r=2 SFT | 3 (h0 + h1 + final) | 2 | 4.0 | 4h |
| r=3 SFT | 4 (h0..h2 + final) | 3 | 5.3 | 5.3h |
| r=5 SFT | 6 (h0..h4 + final) | 5 | 8.0 | 8h |
| r=6 SFT (Phase 20 S1) | 7 | 6 | 9.3 | 9.3h |

**At 5 seeds parallel on 5 GPUs, wallclock = same as 1 GPU.** Per-seed
GPU-hours sum × 5 for total cost.

## Inference cost per query

pass@k requires k samples per prompt. Single sample at max_new=200,
temp=0.8 on A100 ≈ **3.5s** (cold) / **1.2s** (warm batch of 6).

| k | per-query wallclock (warm) | tokens generated |
|---:|---:|---:|
| 1 | 1.2s | ≤200 |
| 5 | 6.0s | ≤1000 |
| 10 | 12.0s | ≤2000 |
| 20 | 24.0s | ≤4000 |

(Single-prompt; batched serving amortizes ~3× better.)

## Combined cost × quality table

| recipe + inference | pass-rate | train GPU-h | inf wallclock | notes |
|---|---:|---:|---:|---|
| base + pass@1 | 0.216 | 0 | 1.2s | trivial reference |
| base + pass@5 | 0.425 | 0 | 6s | pure-inference baseline |
| base + pass@10 | **0.524** | 0 | 12s | strongest no-training |
| r=2 SFT + pass@1 | 0.404 | 4.0 | 1.2s | single-shot strong |
| r=2 SFT + pass@5 | 0.545 | 4.0 | 6s | balanced; recommended ★ |
| r=2 SFT + pass@10 | **0.595** | 4.0 | 12s | training+inference compound |
| r=3 SFT + pass@1 | 0.473 | 5.3 | 1.2s | best single-shot below r=5 |
| **r=3 SFT + pass@5** | **0.567** | 5.3 | 6s | sweet spot ★★ |
| r=3 SFT + pass@10 | 0.591 | 5.3 | 12s | marginal over k=5 |
| r=5 SFT + pass@1 | **0.556** | 8.0 | 1.2s | best pure training |
| r=5 SFT + pass@10 | ~0.65 est | 8.0 | 12s | research-grade |
| r=5 SFT (seed 1 best) | **0.620** | 8.0 | 1.2s | project record single-seed |

## Pareto front

Drawing pass-rate (y) vs total cost (training GPU-h + 100 × inf seconds):

```
0.70 |                                    × r=5+passk10 (est)
0.65 |
0.60 |              ★★ r=3+passk5        × r=2+passk10
0.55 |         ★ r=2+passk5                r=5+passk1
0.50 |    base+passk10
0.45 | base+passk5
0.40 |              r=2+passk1
0.35 |
0.30 |
0.25 | base+passk1
     +--------+--------+--------+--------+-------
       0       5       10      15      20  GPU-h equiv
```

**Pareto optimal**:
1. **base + pass@5** (cheap, ~0.45)
2. **r=2 SFT + pass@5** (balanced, ~0.55) ★
3. **r=3 SFT + pass@5** (best ROI, ~0.567) ★★
4. **r=2 SFT + pass@10** (best balance, ~0.595)

**Dominated** (suboptimal): r=5 SFT + pass@1 (0.556 at 8 GPU-h) is
beaten by r=3 + pass@5 (0.567 at 5.3 GPU-h + 6s). Only useful if
single-shot inference is hard constraint.

## Recommended recipes

### Cheap-deploy (no training infra)
```python
# Inference only. Sample 5 completions, return any that passes verifier.
samples = generate(prompt, k=5, T=0.8, top_p=0.95, max_tok=200)
for s in samples:
    if verify(s): return s
return None
# pass-rate ~0.425, latency ~6s/query
```

### Balanced (single train job, cheap inference)
```python
# One-time: r=2 SFT (4 GPU-h on A100).
model = train_self_improve(base="Qwen2.5-Coder-0.5B", rounds=2,
                            samples=6, train_steps=200)
# Per-query: pass@5 at temp=0.8.
# pass-rate ~0.545, latency ~6s/query
```

### Best-ROI (recommended ★★)
```python
# One-time: r=3 SFT (5.3 GPU-h).
model = train_self_improve(base="Qwen2.5-Coder-0.5B", rounds=3,
                            samples=6, train_steps=200)
# Per-query: pass@5.
# pass-rate ~0.567, latency ~6s/query
```

### Research-max (no compute constraint)
```python
# r=5 SFT (8 GPU-h) + pass@10 at inference.
# pass-rate ~0.65 estimated, latency ~12s/query
# Best single-seed observed: 0.620 (Phase 19 seed 1 r=5 pass@1)
```

## Production checklist

### Reproducibility
- Pin `transformers` + `peft` + `torch` versions in `/tmp/p14_env/`
- Set torch.manual_seed + cuda.manual_seed_all per seed
- Seed-stamped harvest set serialized to JSON (each round)
- LoRA adapter checkpointed per round (PEFT save_pretrained)
- Base model SHA recorded in checkpoint metadata

### Drift monitoring
- Cross-substrate eval (MBPP) at every release — drift detector
- Lift bimodality alert: 5-seed σ > 0.05 indicates init-RNG instability
- Per-challenge pass-rate tracking: saturate vs cold-start vs middle band
- Phase 17 risk #19 trigger: if running r ≥ 4, monitor for plateau/regress

### Eval reproducibility
- `samples=6` is the noise-floor minimum per Phase 16 S3b CLT (σ halves
  from samples=3 → samples=6). For decision-grade eval use samples ≥ 6.
- Use 5 seeds for confidence; never publish single-seed numbers (Risk #14)

### Cost monitoring
- Per-round wall-clock should be 70-90 min on A100. Drift outside this
  band signals tokenizer change, harvest batch size regression, or VRAM
  contention.
- LoRA train ≤ 30s/round. > 60s indicates dataset size regression.

## Cross-substrate numbers (MBPP-100)

For sanity check that recipe transfers:

| recipe | HumanEval | MBPP | Δ |
|---|---:|---:|---:|
| base pass@1 | 0.216 | 0.151 | -0.065 |
| base pass@10 | 0.524 | 0.421 | -0.103 |
| r=2 SFT | 0.404 | 0.353 (P17 SB) | -0.051 |
| r=3 SFT | 0.475 | 0.457 (P18 S3) | -0.018 |
| r=5 SFT (Phase 20 S2 pending) | 0.556 | ??? | TBD |

MBPP is harder substrate; deltas narrow as training compounds (good
sign — recipe transfers, doesn't overfit to HE).

## What this recipe does NOT cover

- **Larger model** (Qwen 1.5B+) — Phase 19 deferred; r=2 SFT signal at
  1.5B was Δ=-0.176 in Phase 9 S4 single test, NOT confirmed at MR
- **Pure RL recipes** — pass@k as reward, not yet implemented
- **Multi-domain mixing** — single-substrate training only
- **Online learning** — corpus update during deployment
- **Tool-use task** (Phase 4 ToolUseArithmeticDomain) — separate recipe

## See also

- `docs/phase19-closeout.md` — saturation curve + deployment optima
- `docs/phase17-closeout.md` — pass@k discovery
- `scripts/phase15_s1/self_improve.py` — training driver
- `scripts/phase17_sa/run_mr_passk.py` — pass@k eval driver
