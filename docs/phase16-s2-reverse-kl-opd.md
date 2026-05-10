# Phase 16 S2 — Reverse-KL OPD is even worse than Forward-KL

Phase 15 S2 retracted forward-KL T=2.0 multi-teacher OPD as
destructive (Δ=−0.088 vs SFT, σ blowup 1.70×, 4/5 seeds destroy
base model). One of the four mechanism candidates was "forward-KL
spreads probability mass uniformly; reverse-KL might fix it" since
DeepSeek V4's actual choice is reverse-KL. S2 here re-runs with
`--kl-direction reverse`, all other hyperparams unchanged.

## Setup

- HumanEval-164, Qwen2.5-Coder-0.5B + LoRA r=16 α=32
- Same 3 specialists (strings/numbers/collections) from
  `checkpoints/phase15_s2/` reused as teachers
- 5 student seeds × OPD-FT against routed teacher logits, T=2.0
- `--kl-direction reverse` (vs Phase 15 S2's `forward`)
- 200 train-steps, AdamW lr=2e-4
- Hardware: GPU 2 (seeds 0/1/2) + GPU 3 (seeds 3/4), ~3.5h wallclock

## Result — even worse than forward-KL

| arm | mean | σ |
|---|---:|---:|
| SFT (Phase 15 S1) | 0.245 | 0.041 |
| Forward-KL OPD (Phase 15 S2) | 0.157 | 0.070 |
| **Reverse-KL OPD** (this commit) | **0.086** | **0.045** |

Per-seed reverse-KL: [0.152, 0.087, 0.102, 0.045, 0.045]

| metric | reverse-KL | forward-KL (P15 S2) | delta |
|---|---:|---:|---:|
| mean | 0.086 | 0.157 | **−0.071** |
| σ | 0.045 | 0.070 | (more concentrated, but at lower mean) |
| Δ vs SFT | −0.159 | −0.088 | **2× more destructive** |
| 2σ_max threshold | 0.090 | 0.140 | — |
| seeds < SFT | 5/5 | 5/5 | — |
| seeds < round-0 | 5/5 | 4/5 | reverse-KL also destroys seed 2 |
| 5/5 seeds → catastrophic? | 4/5 (≤ 0.10) | 1/5 | reverse-KL pushes more seeds to mode collapse |

### Per-seed comparison vs forward-KL on same seeds

| seed | r0 | fwd-KL final | rev-KL final | rev-fwd Δ |
|---:|---:|---:|---:|---:|
| 0 | 0.213 | 0.195 | 0.152 | −0.043 |
| 1 | 0.211 | 0.148 | 0.087 | −0.061 |
| 2 | 0.224 | **+0.018** (only positive seed) | 0.102 | broken |
| 3 | 0.234 | 0.053 | 0.045 | −0.008 |
| 4 | 0.207 | 0.144 | 0.045 | −0.099 |

Reverse-KL **erases the only positive seed** (seed 2) that forward-KL
had. 5 out of 5 reverse-KL seeds end below 0.16; 4 out of 5 below 0.11.

## Mechanism — why reverse-KL is worse

DeepSeek V4 chose reverse-KL [KL(student || teacher)] specifically
to avoid forward-KL's mode-spreading. The expected mode behavior:

- **Forward-KL** [KL(teacher || student)] — student must put mass
  wherever teacher does ("mode-covering"). Spreads student
  distribution.
- **Reverse-KL** [KL(student || teacher)] — student avoids putting
  mass where teacher doesn't ("mode-seeking"). Concentrates student
  on a few teacher modes.

The expected DeepSeek-V4-style benefit: reverse-KL keeps generations
sharp by mode-seeking, avoiding forward-KL's tendency to soften.

What actually happens at our scale: **mode-seeking on noisy
specialist teachers concentrates the student into the teacher's
worst modes**. The strings specialist's catastrophic-then-recovered
training (0.202 → 0.096 → 0.283) leaves residual noise in the
recovered logits. Reverse-KL's mode-seeking inherits this — the
student locks onto the teacher's mode at temperature 2.0, but those
modes happen to include some noise modes the specialist had during
forgetting. Forward-KL spreads across modes (some of them OK), so
its effect was milder.

This is a **specialist-quality-dependent effect**. The S2 result
mirrors the Phase 14 C3 hybrid DPO finding: hybrid did achieve r=1
peak 0.865 on seed 0 (best lift), but had σ blowup because some seeds
collapsed. Reverse-KL has the analogous pattern: mode-seeking can
amplify good teachers, but noisy teachers make it pathological.

## Verdict — OPD as a primary training signal at LoRA scale fails

Together with Phase 15 S2's forward-KL LOSS, this commit closes the
"OPD KL direction matters" question:

- **Forward-KL T=2.0**: LOSS Δ=−0.088 (P15 S2)
- **Reverse-KL T=2.0**: LOSS Δ=−0.159 (this commit)

KL direction doesn't fix OPD at small-LoRA scale. Both directions
destabilize student training, with reverse-KL being more so when
specialists are noisy.

The remaining mechanism candidates (Phase 15 S2 doc):
- Specialist noise (strings spec went 0.202→0.096→0.283) — strongly
  implicated. Reverse-KL's mode-seeking amplifies it.
- Hinton 2015 T² scaling missing — possible minor effect, but
  doesn't explain the 70% σ blowup or per-seed catastrophes.
- "OPD without SFT anchor" — strongly implicated. Without an SFT
  loss term grounding to verifier-passed completions, KL alone has
  no chosen-pair anchor. **Hybrid OPD+SFT is the natural next test**.

## Decision impact

- **Reverse-KL OPD definitively retracted** at LoRA self-improve
  scale. Don't try as drop-in for SFT.
- **Phase 14 C4 deferral upheld 2× over** — both KL directions fail.
- **Hybrid OPD+SFT test** moves up the Phase 16/17 priority list as
  the most-promising remaining OPD-style intervention.
- Risk #18 reaffirmed: naive offline OPD destabilizes small-LoRA
  loops; KL direction CHOICE doesn't rescue it.

## Reproducing

```bash
bash scripts/phase16_s2/run_revkl_a.sh 2  # GPU 2, seeds 0/1/2
bash scripts/phase16_s2/run_revkl_b.sh 3  # GPU 3, seeds 3/4
/tmp/p14_env/bin/python scripts/phase16_s2/analyze.py
```

## See also

- `docs/phase15-s2-opd-results.md` — original forward-KL OPD LOSS
- `docs/phase16-design.md` — Phase 16 plan including this re-test
- `scripts/phase15_s2/self_improve_opd.py` — OPD trainer (reused)
- `scripts/phase14_c4/opd.py` — opd_loss (forward + reverse)
- DeepSeek V4 technical report — reverse-KL design rationale
