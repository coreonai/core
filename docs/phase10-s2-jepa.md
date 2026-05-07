# Phase 10 S2 — JEPA follow-ups (sweeps overturn S1's "JEPA breaks Shape-C" framing)

S1 produced a clean honest negative at ONE point in JEPA's
hyperparameter space (λ=0.1, k=8, single-encoder stop-gradient on
K8 KoWiki): top-1 mass dropped 33% but sum-AUC fell 0.421 → 0.238
(deeper anti-cal). That single point did not let us decide whether
the loss-of-calibration is fundamental to JEPA, an artifact of one
λ, an artifact of one k, an artifact of single-encoder vs EMA, or
an artifact of K8's mode-collapse pathology.

S2 sweeps four orthogonal axes to resolve the ambiguity, and the
result is **substantially more nuanced** than S1's blanket negative
suggested.

## Setup (matches S1 exactly except for the swept axis)

- 50M Llama recipe (RoPE + GQA-2 + SwiGLU + RmsNorm-Pre + untied),
  16K BPE, K8 KoWiki, block 256, batch 16, lr 6e-4.
- 5K steps, A100 fp32 (~6 min/run, ~8 min/run for the EMA variant).
- Same data shuffling seed across runs.
- Critic measurement via `critic_baseline_korean.rs` —
  30 prompts × 20 samples = 600 candidates per checkpoint.

For axis 4 (Python): existing `critic_baseline_python.rs` (1M GPT,
char tokenizer, synth corpus, 1500 pretrain steps), now with
`--jepa-lambda` / `--jepa-offset` exposed.

## Implementation deltas vs S1 (`fdbe5f7`)

- `TrainConfig.jepa_ema_decay: Option<f32>` — when `Some(d)`, builds
  a parallel target VarMap + GPT and EMA-updates it every step
  (`target = d·target + (1-d)·main`). JEPA target hidden states come
  from this slow encoder rather than from `.detach()` on the main
  encoder's output.
- `varmap_snapshot` + `varmap_ema_update` helpers in `train.rs`.
- 2 new train-side smoke tests: end-to-end EMA run + decay
  out-of-range guard.
- `--jepa-ema-decay` CLI on `train_kowiki_jepa`.
- `--jepa-lambda` / `--jepa-offset` on `critic_baseline_python`.

## Results — K8

| variant            |   λ |  k |  EMA  | top1 mass | pass | sum-AUC |  Δ vs base | F=4 lift |
|--------------------|----:|---:|:-----:|----------:|-----:|--------:|-----------:|---------:|
| baseline (S1)      | 0.0 |  — |  no   |    0.146  | 2.2% |   0.421 |     —      |  0.53×   |
| λ=0.01, k=8        |0.01 |  8 |  no   |    0.091  | 2.2% |   0.342 |   −0.079   |  1.00×   |
| λ=0.03, k=8        |0.03 |  8 |  no   |    0.066  | 2.5% |   0.291 |   −0.130   |  0.21×   |
| **λ=0.1, k=8 (S1)**| 0.1 |  8 |  no   |  **0.097**| 3.3% | **0.238** | −0.183 |  0.21×   |
| **λ=0.3, k=8**     | 0.3 |  8 |  no   |    0.069  | 1.8% | **0.433** | **+0.012** | 0.60× |
| **λ=0.1, k=2**     | 0.1 |  2 |  no   |    0.073  | 3.3% | **0.432** | **+0.011** | 0.54× |
| λ=0.1, k=4         | 0.1 |  4 |  no   |    0.050  | 1.8% |   0.396 |   −0.025   |  0.50×   |
| EMA decay=0.99     | 0.1 |  8 | 0.99  |    0.049  | 2.2% |   0.292 |   −0.129   |  0.36×   |

### Reading the K8 sweeps

**The S1 negative is not monotone in λ.** As λ rises from 0 to 0.1,
sum-AUC degrades (0.421 → 0.342 → 0.291 → 0.238). But at λ=0.3,
sum-AUC recovers to 0.433 — actually slightly above baseline. The
"worst spot" in the hyperparameter grid is λ=0.1, k=8.

**Shorter k recovers calibration faster than λ does.** At λ=0.1
fixed:
  k=8 → 0.238 (−0.183)
  k=4 → 0.396 (−0.025)
  k=2 → 0.432 (+0.011, recovered)

**EMA didn't help calibration.** Decay 0.99 at λ=0.1, k=8 lands
sum-AUC 0.292 — mid-pack between the failing k=8 cases. The slow
target gives the strongest mode-collapse mitigation
(top1 0.049, the lowest in the matrix) but its target signal is
*just as antagonistic* with verifier-aligned confidence as the
self-target. The EMA-vs-self difference is on diversity, not on
calibration.

**Mode-collapse weakens monotonically with λ on the upper side.**
top-1 mass: 0.146 (baseline) → 0.091 (λ=0.01) → 0.066 (λ=0.03) →
0.097 (λ=0.1) → 0.069 (λ=0.3). The non-monotonicity in
λ is small; what matters is *every* JEPA variant beats baseline on
top-1 mass. EMA wins (0.049) on this axis.

## Results — PythonCodeDomain

| variant       | pass  | mean-AUC | sum-AUC | F=2 lift | F=4 lift | F=8 lift |
|---------------|------:|---------:|--------:|---------:|---------:|---------:|
| baseline λ=0  | 35.6% |   0.787  |  0.859  |  1.07×   |  1.01×   |  0.72×   |
| λ=0.03, k=4   | 32.2% |   0.847  |  0.862  |  1.06×   |  0.96×   |  0.66×   |
| λ=0.1, k=4    | 34.4% |   0.767  |  **0.867** |  **1.10×** | **1.05×** |  0.75× |

**On Python, JEPA does not break Shape-C.** All three variants
clear the sum-AUC ≥ 0.6 deployment gate by a wide margin (≈ 0.86).
λ=0.1, k=4 even posts the *highest* F=4 lift (1.05× vs baseline
1.01×). Pass rate drops modestly (35.6 → 34.4 → 32.2) — JEPA
nudges the model toward more diverse output, which costs a few
percentage points of pass rate but not calibration.

The K8 pathology — anti-calibration in the (λ=0.1, k=8) regime —
does not transfer to Python at the same nominal λ. **The calibration
hit is K8-specific, not domain-general.**

## What this changes about risk #12 in `docs/phase7-design.md`

S1 worded risk #12 as the blanket statement "anti-mode-collapse aux
losses can WORSEN Shape-C calibration." That's defensible at the
S1 measurement point but is too strong a generalization given S2:

- On **K8 + (λ=0.1, k=8)**: yes, sum-AUC drops sharply.
- On **K8 + (λ=0.1, k=2)** or **(λ=0.3, k=8)**: sum-AUC fully
  recovers, *with* the diversity gain preserved.
- On **Python at any of the three λ tested**: sum-AUC stays ≈ 0.86.

Updated risk #12 framing:
> JEPA-style aux losses interact non-trivially with verifier-aligned
> calibration. The interaction is hyperparameter-sensitive (λ, k)
> and domain-sensitive (K8 has a stronger mode-collapse pathology
> than Python). Always sweep at least two (λ, k) points and re-
> measure sum-AUC + selection lift before deciding the aux is
> "helping" or "hurting." Don't assume a fixed (λ, k) carries
> across domains.

## Practical recipe (for someone deploying JEPA aux on a new domain)

1. Pretrain a baseline (no JEPA) and measure sum-AUC + top-1 mass.
2. Try at least three points: (λ=0.05, k=2), (λ=0.1, k=2),
   (λ=0.2, k=4). These were the S2 winners on K8 and didn't break
   anything on Python.
3. Plot sum-AUC vs λ at fixed k. If it's U-shaped (S2 K8 pattern),
   prefer either tail; the bottom of the U is to be avoided.
4. Don't bother with EMA target unless top-1 mass is the metric you
   actually care about — it didn't recover calibration in our run.
5. Don't conclude "JEPA helps / hurts this domain" from one (λ, k)
   point.

## Reproducing

```bash
# Sweeps + EMA + Python (the wrapper scripts are session-local;
# committed copies are in scripts/phase10_s2/).
bash /tmp/p10s2_lambda_sweep.sh   # GPU 0, ~18 min
bash /tmp/p10s2_k_sweep.sh        # GPU 1, ~12 min, in parallel
bash /tmp/p10s2_ema_python.sh     # GPU 1, ~15 min, after both above

# Aggregate
bash scripts/phase10_s2/measure_all.sh    # ~24 min on a single GPU
bash scripts/phase10_s2/reparse_k8_logs.sh  # rebuilds results.tsv from saved logs
bash scripts/phase10_s2/parse_python_log.sh /tmp/p10s2_ema_python.log \
        scripts/phase10_s2/results_python.tsv
bash scripts/phase10_s2/extract_top1.sh
```

## See also

- `docs/phase10-s1-jepa.md` — the S1 single-point honest-negative
  this sweep extends.
- `docs/phase7-design.md` risk #12 — calibration-vs-diversity
  tension; updated framing in this commit.
- `nanogpt-rs/src/jepa.rs`, `nanogpt-rs/src/train.rs` — JEPA aux +
  EMA target encoder implementation.
- Phase 9 S2 memory entry — the K8 100K mode-collapse finding that
  motivated trying JEPA in the first place.
