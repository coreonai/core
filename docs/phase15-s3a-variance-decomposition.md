# Phase 15 S3a — variance decomposition at Phase 14 substrate

Phase 14 S1 measured σ=0.011 across 5 seeds, but the seed flag
controlled BOTH the LoRA delta_A init RNG AND the harvest sampling
RNG. The σ=0.011 measurement was acknowledged as a *lower bound*
since real noise spans multiple axes. S3a separates init-RNG from
harvest-RNG to dissect that bound.

## Setup

- Same Phase 14 substrate (Qwen2.5-Coder-0.5B + LoRA r=16 α=32, 25
  HumanEval-style problems)
- New harness: `scripts/phase15_s3/decompose_seeds.py` exposes
  `--init-seed` and `--harvest-seed` as separate flags
- 5 init runs (vary --init-seed, fix --harvest-seed=0) → σ_init
- 5 harvest runs (--init-seed=0, vary --harvest-seed) → σ_harvest
- Phase 14 S1's existing 5-seed runs serve as σ_combined (paired)
- Hardware: GPU 0 (init-axis) + GPU 1 (harvest-axis), ~45 min
  parallel wallclock

## Result

| axis | n | mean | σ | per-seed |
|---|--:|---:|---:|---|
| init-only | 5 | 0.853 | **0.004** | [0.850, 0.860, 0.850, 0.855, 0.850] |
| harvest-only | 5 | 0.852 | **0.016** | [0.850, 0.880, 0.845, 0.845, 0.840] |
| combined (S1) | 5 | 0.851 | **0.011** | [0.850, 0.870, 0.845, 0.850, 0.840] |

### Decomposition

σ²-share:

| axis | σ² | share |
|---|---:|---:|
| init | 0.004² = 0.000016 | **7%** |
| harvest | 0.016² = 0.000256 | **93%** |

Phase 14 substrate is **harvest-dominated**.

### Additivity

If the two axes were independent:
> σ_combined ≈ √(σ_init² + σ_harvest²) = √(0.004² + 0.016²) = **0.017**

Observed σ_combined = **0.011**. Predicted/observed ratio = **1.46**.

Combined σ is *lower* than the independent prediction → the paired-
seed scheme's matched RNG produces seeds that are mildly *anti-
correlated* across axes (a fixed pairing of init and harvest seeds
covers a smaller portion of the noise volume than independent
sampling would). Phase 14's σ=0.011 underestimated the noise
volume by ~46% versus a fully independent measurement.

## Implications

### For Phase 14 retracted claims (C2 Muon, C3 DPO)

C2 and C3 used paired-seed comparisons (matched init+harvest between
Muon vs AdamW, hybrid DPO vs SFT). Because the noise was matched
between arms, the **retractions remain valid even though the σ=0.011
floor was an underestimate**. The 2σ=0.022 threshold used at C2 was
thus appropriate for paired-seed comparison; an unpaired Muon-vs-
AdamW measurement would have needed 2σ=0.034 (independent prediction).

### For future single-arm noise reporting

When reporting "σ across N seeds" without specifying axis, the result
depends heavily on whether init+harvest are paired (most common) or
independent. **Paired is the conservative measurement for paired-
arm comparisons but underestimates absolute substrate noise.**

### For Phase 15 substrate (HumanEval-164)

Phase 15 S1's mechanism analysis suggested **σ_init might dominate**
at HumanEval scale because LIFTED/FLAT seed groups had similar
harvests (Jaccard 0.52-0.56) but radically different LoRA-FT
trajectories.

S3a's Phase 14 result is the opposite — σ_harvest dominates. Possible
explanations:

1. Phase 14's saturated substrate (84% saturated) limits how much
   LoRA init can move outcomes; trained models converge to similar
   solutions regardless of init.
2. Phase 15's harder substrate (44% cold-start, 52% headroom)
   amplifies init differences because the model has more room to
   diverge.

Decomposing Phase 15 (S3b) is the natural follow-up. If S3b shows
σ_init >> σ_harvest at HumanEval, that confirms substrate-shape
dependence and recommends **multi-init averaging** as a noise-
reduction technique for Phase 15 algorithmic comparisons.

## What this commit changes

- Adds `docs/phase15-s3a-variance-decomposition.md` (this file)
- Adds Phase 14 substrate σ-decomposition data (10 new runs in
  scripts/phase15_s3/)
- **Does NOT change retroactive verdict on C2/C3** (paired comparison
  still valid)

## See also

- `docs/phase14-s1-substrate.md` — original Phase 14 σ=0.011
  measurement (acknowledged as lower bound)
- `docs/phase15-s1-substrate.md` — Phase 15 substrate (HumanEval)
  qualification + mechanism finding (LIFTED/FLAT bimodality)
- `scripts/phase15_s3/{decompose_seeds.py, run_init_axis.sh,
  run_harvest_axis.sh, analyze.py}` — decomposition harness
- Future: S3b at HumanEval substrate (init/harvest split + temperature
  axis + checkpoint axis if budget permits)
