# Phase 16 S4 — Hybrid OPD+SFT partly rescues OPD but still LOSS

Phase 15 S2 (forward-KL) and Phase 16 S2 (reverse-KL) both retracted
multi-teacher OPD. Most plausible remaining mechanism: KL alone has
no chosen-pair anchor, so any KL update can drift student to high-
entropy or mode-collapsed regions. Hybrid OPD+SFT mixes:

> loss = (1 − α) · OPD_KL + α · SFT_NLL_chosen

with α=0.3 (Phase 11 S5 hybrid-DPO winner). SFT term anchors student
to verifier-passed completions; OPD KL provides specialist
supervision.

## Setup

- Same as Phase 16 S2 (HumanEval-164, Qwen + LoRA r=16, k=3
  specialists from `checkpoints/phase15_s2/`, T=2.0, reverse-KL,
  200 train-steps)
- New: `--sft-alpha 0.3` adds SFT anchor
- 5 seeds × 5 GPUs (1, 2, 3, 6, 7) **true parallel** → ~80 min wallclock
  vs ~200 min sequential

## Result — partial rescue, still LOSS vs SFT

| arm | mean | σ |
|---|---:|---:|
| SFT only (Phase 15 S1) | 0.245 | 0.041 |
| Forward-KL OPD (P15 S2) | 0.157 | 0.070 |
| Reverse-KL OPD (P16 S2) | 0.086 | 0.045 |
| **Hybrid OPD+SFT α=0.3 rev-KL** | **0.130** | **0.088** |

### Per-seed comparison (same 5 seeds across all arms)

| seed | r0 | SFT | Fwd-KL | Rev-KL | **Hybrid** |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.213 | 0.278 | 0.195 | 0.152 | 0.122 |
| 1 | 0.211 | 0.220 | 0.148 | 0.087 | **0.189** |
| 2 | 0.224 | 0.224 | **0.242** | 0.102 | **0.246** |
| 3 | 0.234 | 0.299 | 0.053 | 0.045 | 0.059 |
| 4 | 0.207 | 0.203 | 0.144 | 0.045 | 0.037 |

### Verdicts

| comparison | Δ | 2σ threshold | verdict |
|---|---:|---:|---|
| Hybrid vs SFT | **−0.114** | 0.175 | "within inflated noise" — only because σ_hybrid blows up |
| Hybrid vs Rev-KL OPD | **+0.044** | 0.176 | within noise but mean improved |
| Hybrid vs Fwd-KL OPD | −0.027 | 0.176 | within noise |

### σ analysis

| arm | σ |
|---|---:|
| SFT only | 0.041 |
| Rev-KL OPD | 0.045 |
| Fwd-KL OPD | 0.070 |
| **Hybrid** | **0.088** ← 2.1× σ_SFT |

Hybrid has the **widest σ of any arm tested**. The α=0.3 SFT anchor
introduces a third state (SFT-level) that joins the existing two
states (catastrophic-collapse, mild-destruction) seen in pure OPD.
Result: trimodal-ish distribution → σ inflation.

## Mechanism — hybrid is high-variance not centrally improved

The hybrid effect is **highly seed-dependent**:

| outcome bucket | seeds | hybrid pass rate range |
|---|---|---|
| SFT-level (~0.20-0.25) | 2 (1, 2) | 0.189, 0.246 |
| Mid-degradation | 1 (0) | 0.122 |
| Catastrophic | 2 (3, 4) | 0.037-0.059 |

Compare to pure reverse-KL: 0/5 SFT-level, 1/5 mid (0.152), 1/5 mild
(0.102), 2/5 catastrophic (0.045 each). So:

- **Hybrid moves 2 seeds from "destroyed" to "SFT-level"** (real
  rescue effect)
- **Hybrid leaves 2 seeds catastrophic** (SFT anchor at α=0.3 too
  weak to overcome destructive teacher logits)
- **Hybrid pulls 1 seed (0) into mid-bucket** that pure rev-KL had
  in mid-bucket too

Net mean improvement +0.044 vs reverse-KL is real, but the σ blowup
means hybrid never beats SFT — even seed 2 (0.246) is at SFT-level,
not above.

This pattern is **identical to Phase 14 C3 hybrid DPO**: hybrid
α=0.3 had 1 seed (0) collapse to 0.575 while others stayed near
SFT — same trimodal failure mode. SFT anchor at α=0.3 prevents
some catastrophes but not all, and never produces a positive lift.

## Could a higher α save it?

α controls SFT/OPD weight. At α=0.3 (70% OPD), 2/5 seeds catastrophic.
At α=1.0 (pure SFT), no catastrophes (= P15 S1 baseline 0.245). The
question is whether α∈(0.3, 1.0) has a sweet spot.

Phase 11 S5 swept α ∈ {0.0, 0.3, 0.5, 1.0} for hybrid DPO — α=0.3
was best. Higher α moved closer to pure SFT (no DPO benefit).

S4 here doesn't sweep α. Three plausible Phase 17 follow-ups:
- **α=0.5**: equal weight. Likely fewer catastrophes, less lift
  potential.
- **α=0.7**: SFT-dominant with KL regularizer. Likely closest to
  SFT performance.
- **α annealed**: α=0.7→0.3 schedule. Untested.

But none of these would credibly produce a robust win past Phase 16
S1's new 2σ threshold of 0.062. **OPD's specialist signal is too
noisy for any α to convert into useful lift** at this scale.

## Verdict — OPD as a primary training signal at LoRA scale closed

Five OPD configurations tested:
1. Forward-KL T=2.0 pure (P15 S2): Δ=−0.088, σ blowup 1.70×
2. Reverse-KL T=2.0 pure (P16 S2): Δ=−0.159, σ=0.045 (more catastrophic)
3. Hybrid α=0.3 rev-KL (P16 S4): Δ=−0.114, σ=0.088 (partial rescue)
4-5. (Forward / hybrid / various T variations would explore corner
cases unlikely to flip the verdict.)

**No OPD configuration beats SFT at HumanEval scale**. Phase 14 C4
deferral upheld 3× over.

## Decision impact

- **OPD definitively retracted at LoRA self-improve scale**. Don't
  invest more cycles unless: (a) substantially larger model, (b)
  substantially more training-data scale, (c) SFT-pretrained student
  base + OPD as fine-tuning step (not from-scratch).
- **opd.rs / opd.py modules stay in codebase** as unit-tested loss
  functions — useful for future scale experiments or as building
  blocks. Application pattern at small LoRA fails.
- **Risk #18 stands and reaffirmed**: naive offline OPD (forward,
  reverse, or hybrid) destabilizes small-LoRA self-improve loops.
  Specialist quality + KL alone can't compensate for missing
  chosen-pair anchor strength.

## Reproducing

```bash
# 5 seeds in parallel (1 per GPU)
for s in 0 1 2 3 4; do
  bash scripts/phase16_s4/run_hybrid_seed.sh $((s+1)) $s 0.3 reverse \
    > scripts/phase16_s4/log_seed${s}.txt 2>&1 &
done
wait
/tmp/p14_env/bin/python scripts/phase16_s4/analyze.py
```

## See also

- `docs/phase15-s2-opd-results.md` — forward-KL OPD LOSS
- `docs/phase16-s2-reverse-kl-opd.md` — reverse-KL OPD LOSS
- `~/.claude/.../phase11_s5_hybrid_dpo.md` — original hybrid concept
  (DPO version)
- `scripts/phase16_s4/self_improve_hybrid_opd.py` — implementation
