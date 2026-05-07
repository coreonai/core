# Phase 10 S2 — JEPA follow-ups (λ sweep, k sweep, EMA target, Python domain)

Phase 10 S1 found that JEPA-aux (λ=0.1, k=8) on K8 reduced top-1 mass
by 33% but **worsened** sum-AUC from 0.421 → 0.238 (risk #12 in
`docs/phase7-design.md`). S2 explores four orthogonal axes to find
out whether that loss-of-calibration is fundamental, a hyperparameter
artifact, or a domain artifact.

## Four follow-up axes

1. **λ sweep on K8**: λ ∈ {0.01, 0.03, 0.1, 0.3} at k=8.
   Hypothesis: a lower λ might recover calibration while keeping
   diversity gains.
2. **k sweep on K8**: k ∈ {2, 4, 8} at λ=0.1.
   Hypothesis: shorter prediction horizon is less disruptive to
   token-level CE alignment.
3. **EMA target encoder on K8**: separate slow encoder (BYOL/I-JEPA
   style) at decay=0.99, λ=0.1, k=8.
   Hypothesis: stop-gradient noise from a self-target may be hurting
   calibration; a slow target gives a more stable signal.
4. **JEPA on PythonCodeDomain**: Phase 9 S5 setup, with and without
   λ=0.1, k=4 JEPA aux during pretraining.
   Hypothesis: K8's `\n`-mode pathology is Korean-specific; on a
   pass-rate-headroom domain (Python) JEPA's diversity might *help*
   calibration instead of hurting it.

## Setup

Same model architecture as S1 (50M Llama recipe, 16K BPE, K8 corpus
on the K8 runs; 1M GPT, char tokenizer, synth corpus on Python).
Same training budget (5K steps K8, 1500 steps Python). All runs on
A100 fp32. Each K8 run ≈ 6 min, each Python run ≈ 1 min plus 3 min
for AUC measurement.

## Reproducing

```bash
# 1. λ sweep (GPU 0)
bash /tmp/p10s2_lambda_sweep.sh

# 2. k sweep (GPU 1)
bash /tmp/p10s2_k_sweep.sh

# 3. EMA + Python (GPU 1, after sweeps)
bash /tmp/p10s2_ema_python.sh

# 4. Aggregate sum-AUC + selection lift across all K8 checkpoints
bash scripts/phase10_s2/measure_all.sh
```

The wrapper scripts at `/tmp/` are the exact incantations used in
this session; they reference checkpoint paths and tokenizers that
exist in the repo's gitignored `checkpoints/` and `data/` trees.

## Result (K8 sweeps)

(See `results.tsv` after running `measure_all.sh`.)

| variant            |   λ |  k | pass | mean-AUC | sum-AUC | F=4 lift |
|--------------------|----:|---:|-----:|---------:|--------:|---------:|
| baseline (S1)      | 0.0 |  — | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.01, k=8        |0.01 |  8 | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.03, k=8        |0.03 |  8 | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.1, k=8 (S1)    | 0.1 |  8 | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.3, k=8         | 0.3 |  8 | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.1, k=2         | 0.1 |  2 | TBD  |    TBD   |   TBD   |    TBD   |
| λ=0.1, k=4         | 0.1 |  4 | TBD  |    TBD   |   TBD   |    TBD   |
| EMA decay=0.99     | 0.1 |  8 | TBD  |    TBD   |   TBD   |    TBD   |

(Filled in by `measure_all.sh`.)

## Result (Python domain)

| variant       | pass | mean-AUC | sum-AUC | F=4 lift |
|---------------|-----:|---------:|--------:|---------:|
| baseline λ=0  | TBD  |   TBD    |   TBD   |    TBD   |
| λ=0.03, k=4   | TBD  |   TBD    |   TBD   |    TBD   |
| λ=0.1, k=4    | TBD  |   TBD    |   TBD   |    TBD   |

## Reading the results

(Filled in after measurement.)

## See also

- `docs/phase10-s1-jepa.md` — the S1 baseline result this extends.
- `docs/phase7-design.md` risk #12 — the calibration-vs-diversity
  tension this is testing.
