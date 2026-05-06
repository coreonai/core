# Phase 9 S4 — external HF model validation

Validates the Phase 7 design-doc decision tree on a model that
wasn't trained in this repo. The Rust `critic_baseline_*.rs`
examples all use 1M-param in-house GPTs; this checks whether the
sum-AUC ≥ 0.6 deployment gate, the F≤8 cap, and the
"anti-calibration on undertrained / over-fit models" risk also
hold for a 0.5–1.5B HF model with its own BPE.

## Setup

```bash
python3 -m venv /tmp/s4_env
/tmp/s4_env/bin/pip install torch transformers accelerate
```

(GPU ~3GB free is enough for the 0.5B variant; 1.5B fits in ~6GB.)

## Run

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/s4_env/bin/python harvest.py \
    "Qwen/Qwen2.5-Coder-0.5B" 32 harvest_coder_0p5b.json

/tmp/s4_env/bin/python analyze.py harvest_coder_0p5b.json
```

The harvest:
1. Loads the HF model in fp16 on GPU.
2. For each of 6 challenges (3 from `PythonCodeDomain` + 3 simpler
   arithmetic completions), samples 32 stochastic continuations,
   truncates each at the first newline (mirroring the in-house
   Generator), and records (mean log-prob, sum log-prob) over the
   kept tokens.
3. Verifies via `python -c` on `prompt + completion + suffix` and
   labels each sample.
4. Dumps JSON.

The analyzer:
1. Computes sum-AUC, mean-AUC, random-AUC.
2. Reports per-challenge pass rates.
3. Runs the Phase 7 decision tree (`>=0.6 deploy`, `[0.5, 0.6)
   marginal`, `[0.4, 0.5) no-signal`, `<0.4 anti-cal`).
4. Reports F-sweep selection lift (1, 2, 4, 8, 16) for direct
   comparison to the in-house critic\_baseline matrix.

## Results (committed snapshot)

| Model                | params | pass | mean-AUC | sum-AUC | F=4 lift | Verdict |
|----------------------|-------:|-----:|---------:|--------:|---------:|---------|
| Qwen2.5-Coder-0.5B   |   494M | 9.9% |    0.502 | **0.702** | **1.87×** | PASS |
| Qwen2.5-Coder-1.5B   |  1.54B | 6.8% |    0.232 |   0.474 |    0.64× | NO SIGNAL |

### What this confirms

- **Sum-AUC gate generalizes.** 0.5B-Coder lands cleanly on the PASS
  side of the 0.6 threshold and selection lift (1.95× at F=8) is the
  strongest in the whole matrix — beating in-house K9 (1.22×), P8
  (1.00×), and P9 multi-assert (1.32×).
- **Mean-vs-sum split holds.** Mean-AUC is at chance (0.502) on
  this length-varying domain (slot completions vary 1–24 tokens);
  only sum captures the verifier-aligned signal. Same as Phase 7 S2.

### What this overturns

- **"More capacity ⇒ better calibration"** is wrong here too. Phase
  7 S2 implied bigger pretrain helps; Phase 9 S2 already showed K8
  100K was worse than 30K. Phase 9 S4 confirms it on an external
  model: 1.5B-Coder is worse than 0.5B-Coder (sum-AUC 0.474 < 0.702,
  selection lift drops below 1.0 at every F ≥ 2).

  Mechanism: 1.5B has stronger priors on common patterns
  (`s = 0`, `def f(): return 1`), and these priors drown out the
  rare correct completions (`"hello"`, `5`). Confidence ↑ on the
  *wrong* answer = anti-calibration approaches 0.5 from above.

### What this means for deployment

For a candidate Shape C target, run this exact harvest on a
representative model. If sum-AUC ≥ 0.6 and selection lift ≥ 1.10×
at F=2 or F=4, deploy. If lift drops below 1.0, the model's priors
have over-fit relative to the verifier — train less, use a smaller
model, or skip Shape C for this domain.

## See also

- `docs/phase7-design.md` — the operational guide and decision tree.
- `llm-actors/examples/critic_baseline_python.rs` — the in-house
  PythonCodeDomain measurement (the matrix row this extends).
- Memory entries `phase9_s2_*` and `phase7_session1_*` for prior
  K8 / Arithmetic falsifier tests.
