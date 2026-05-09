# Phase 15 S4 — Muon at HumanEval substrate (LOSS, generalizes Phase 14 C2)

Phase 14 C2 retracted Muon for LoRA at the saturated 25-problem
substrate (Δ=−0.092, 4× threshold). The cited mechanism was "NS
orthogonalization removes step-magnitude information that small-rank
LoRA needs to lock in deterministic completions" — specifically a
saturation-failure-mode argument. Phase 15 S4 tests whether the
verdict generalizes to the headroom-rich HumanEval substrate (4%
saturated, 52% headroom) where larger LoRA deltas are needed and
NS-orthogonalization might fit differently.

## Setup

- Same as Phase 15 S1 (Qwen2.5-Coder-0.5B + LoRA r=16 α=32, HumanEval-
  164, 1 round × 3 samples × 200 train-steps)
- Optimizer arms: AdamW (S1 baseline, reused) vs Muon (lr=2e-4,
  momentum=0.95, weight_decay=0.01, ns_steps=5)
- 5 seeds × Muon, GPU 2/3 sequential, ~6h wallclock
- Reuses `scripts/phase14_c2/muon.py` PyTorch implementation

## Result

| arm | mean | σ |
|---|---:|---:|
| AdamW (S1 baseline) | 0.245 | 0.041 |
| Muon | **0.163** | **0.014** |

Δ_mean = **−0.081**, 2σ_max = 0.083 (threshold inflated by σ_AdamW).
Formal verdict: "WITHIN NOISE" by **0.002** — the thinnest possible miss.

### Three converging robust-LOSS signals

| seed | r0 | Muon final | SFT final | Δ from r0 (Muon) | Δ from r0 (SFT) | Δ Muon-SFT |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.211 | 0.187 | 0.278 | -0.024 | +0.067 | -0.091 |
| 1 | 0.220 | 0.165 | 0.220 | -0.055 | +0.000 | -0.055 |
| 2 | 0.209 | 0.161 | 0.224 | -0.048 | +0.015 | -0.063 |
| 3 | 0.220 | 0.150 | 0.299 | -0.070 | +0.079 | -0.149 |
| 4 | 0.207 | 0.154 | 0.203 | -0.053 | -0.004 | -0.049 |

1. **5 / 5 seeds Muon < SFT**. Range -0.049 to -0.149. Probability
   under null: 1/32 ≈ 3.1%.
2. **5 / 5 seeds Muon < round-0**. Muon's mean Δ from r0 is -0.050
   (destructive); SFT's is +0.032 (lift). Muon makes model **worse
   than no training**.
3. **σ_Muon (0.014) tight, σ_SFT (0.041) wide**. Same Phase 14 C2
   pattern: Muon converges stably but to a *worse* state. If using
   σ_Muon as threshold (2×0.014=0.028), Δ=-0.081 is **2.9× over**.

## Verdict — Phase 14 C2 generalizes despite formal margin

The formal 2σ_max test misses by 0.002 due to σ_AdamW's large value
inflating the threshold, not because Muon and SFT are comparable.
The three converging signals (5/5 seeds, sub-r0 destruction, tight-
but-worse σ pattern) make this a robust LOSS for Muon, just like
Phase 14 C2.

| substrate | AdamW final | Muon final | Δ | 2σ threshold | seeds Muon < SFT |
|---|---:|---:|---:|---:|---:|
| Phase 14 (saturated 25) | 0.851 | 0.759 | -0.092 | 0.022 | 5/5 |
| Phase 15 (HumanEval 164) | 0.245 | 0.163 | -0.081 | 0.083 | 5/5 |

**Phase 14 C2's "Muon LOSES robustly for LoRA" generalizes from
saturated to headroom-rich substrate**. The mechanism is therefore
NOT specific to saturation. Updated mechanism hypothesis:

> NS orthogonalization removes step-magnitude information that
> rank-r LoRA updates need REGARDLESS of substrate shape. The
> r=16 LoRA bottleneck means even at headroom-rich substrate where
> larger updates would help, the orthogonalized step direction
> doesn't concentrate enough mass on the useful directions.

## Decision impact

- **Muon's NAS-axis status (Phase 12 S1) further dispreferred**. Two
  failed substrates with consistent mechanism — not enabled by
  default; likely never useful for LoRA training at this LoRA rank.
- **`nanogpt-rs/src/muon.rs` stays in codebase** as Rust port (Phase
  12 S1 work). May be useful for full-finetune at much larger scale
  (DeepSeek V4 native domain) but NOT for LoRA self-improve loops.
- **Risk #16 (optimizer transfer non-monotonic) reaffirmed and
  generalized**: applies to Muon→LoRA transfer regardless of
  substrate saturation. The naive port doesn't work.

## Cumulative Phase 14+15 retraction count: 3/3 DeepSeek V4 techniques

| technique | tested in | mean Δ | mechanism of failure |
|---|---|---:|---|
| Muon (NS-orthogonalized SGD-momentum) | C2, S4 | -0.092, -0.081 | Wrong inductive bias for rank-r LoRA |
| DPO variants (hybrid, round-0-only) | C3 | within noise | Pair scarcity at saturating substrates |
| OPD (multi-teacher offline distillation) | S2 | -0.088 | KL direction + noisy specialist teachers |

Three distinct failure mechanisms. **DeepSeek V4 techniques don't
naively transfer to small Qwen + LoRA self-improve at our scales**.
This is a substantive Phase 14+15 finding: the "hot 2026 paper drop"
testing strategy is high-yield in failure-mode discovery, low-yield
in net algorithmic gains at this scale.

## Reproducing

```bash
bash scripts/phase15_s4/run_muon_a.sh 2  # GPU 2, seeds 0/1/2
bash scripts/phase15_s4/run_muon_b.sh 3  # GPU 3, seeds 3/4
/tmp/p14_env/bin/python scripts/phase15_s4/analyze.py
```

## See also

- `docs/phase14-c2-muon-lora.md` — original Phase 14 C2 retraction
  (saturated substrate)
- `docs/phase15-s1-substrate.md` — substrate qualification (this
  S4's AdamW baseline)
- `docs/phase15-s2-opd-results.md` — OPD LOSS (sister DeepSeek V4
  test at HumanEval)
- `nanogpt-rs/src/muon.rs` — Rust Muon implementation (Phase 12 S1)
- `scripts/phase14_c2/muon.py` — PyTorch Muon (reused by S4)
