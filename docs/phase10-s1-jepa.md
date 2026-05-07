# Phase 10 S1 — LLM-JEPA aux loss on K8 KoWiki

## Hypothesis

Phase 9 S2 found that K8 Korean training over-fits onto a small set
of high-frequency tokens (`\n`, common particles) when scaled past
~30K steps, with sum-AUC for Shape-C **dropping** as more pretrain
budget piles on (K8 30K → 0.363, K8 100K → 0.307; risk #9 in the
Phase 7 design doc).

LLM-JEPA proposes a fix: alongside next-token CE, predict the
hidden state at offset `+k` (with stop-gradient on the target).
The latent-prediction term should push the model away from the
"emit `\n` on every step" attractor because emitting the same
token cannot match a varying future hidden state.

Falsifiable prediction:
- JEPA-aux **lowers** top-1 softmax mass (less mode collapse).
- JEPA-aux **raises** sum-AUC on KoreanCompletionDomain
  (Shape-C signal recovers).
- Pass rate stays comparable or improves slightly.

## Setup

Two paired runs, identical config except `--jepa-lambda`:

```bash
./target/release/examples/train_kowiki_jepa \
  --data data/kowiki/kowiki_clean.txt \
  --tokenizer data/kowiki/kowiki_bpe.json \
  --steps 5000 --jepa-lambda 0.0 \
  --save checkpoints/p10s1_baseline.safetensors

./target/release/examples/train_kowiki_jepa \
  --data data/kowiki/kowiki_clean.txt \
  --tokenizer data/kowiki/kowiki_bpe.json \
  --steps 5000 --jepa-lambda 0.1 --jepa-offset 8 \
  --save checkpoints/p10s1_jepa01.safetensors
```

- 50M Llama-recipe model (RoPE + GQA-2 + SwiGLU + RmsNorm-Pre +
  untied), 16K BPE, block_size 256, batch 16, lr 6e-4.
- Single A100, ~5 min per run.
- Same data shuffling seed.

Critic measurement via the existing
`llm-actors/examples/critic_baseline_korean.rs` against each
checkpoint — 30 prompts × 20 samples = 600 candidates each.

## Result

| metric                  |    baseline | JEPA λ=0.1 k=8 | direction |
|-------------------------|------------:|---------------:|-----------|
| train_loss              |       7.236 |          7.277 | ~equal |
| **top-1 softmax mass**  |    **0.146** |   **0.097 (−33%)** | ✓ JEPA |
| pass_rate (verifier)    |       2.2 % |    3.3 % (+50%) | ✓ JEPA |
| LogitCritic mean-AUC    |       0.418 |          0.262 | ✗ JEPA |
| **LogitCritic sum-AUC** |   **0.421** |   **0.238 (anti-cal)** | ✗ JEPA |
| F=2 selection lift      |       0.62× |          0.61× | ~equal |
| F=4 selection lift      |       0.54× |          0.21× | ✗ JEPA |
| F=8 selection lift      |       0.44× |          0.03× | ✗ JEPA |
| F=16 selection lift     |       0.14× |          0.00× | ✗ JEPA |

## Reading the result

**Two of three predictions confirmed, the third overturned.**

✓ Mode collapse measurably weaker — top-1 mass drops 33%.
  Sample text (same prompt + seed) shows much wider token diversity.

✓ Pass rate up 50% (relative): the more-diverse model occasionally
  produces a Korean completion the heuristic verifier accepts.

✗ **Shape-C critic gets worse, not better.** Sum-AUC drops from
  0.421 (already a "FAIL" in the Phase 7 decision tree) to 0.238,
  deep into anti-calibration territory. F=4 selection lift halves
  (0.54× → 0.21×); F=16 collapses to 0.00× — the JEPA model's most-
  confident completions are *never* the verifier-accepted ones.

The mechanism is illuminating. JEPA's latent objective rewards
hidden states that **distinguish themselves** from the future, in
order to make next-step prediction feasible. That distinctiveness
is orthogonal to (and partly antagonistic with) verifier-aligned
confidence: the model emits a wider distribution of tokens but its
log-prob ordering is no longer a useful proxy for "will this
sentence parse as Korean?". Diversity ≠ calibration.

Stated differently: Phase 9 S2 showed pretrain-only K8 trades
off "model-relevant pretraining loss" against "verifier-aligned
calibration." JEPA-aux trades off **mode-collapse mitigation**
against **verifier-aligned calibration**. The two trade-offs share
the second arm — Shape-C signal — but pull on the first arm in
*opposite* directions.

## Implications

1. **Risk #12 added to `docs/phase7-design.md`.** Adding a
   representation-quality auxiliary loss can improve diversity /
   pass rate while *worsening* Shape-C calibration. Measure both
   metrics; do not assume one tracks the other.

2. **JEPA is not a free lunch for Shape-C deployment.** If the
   downstream goal is "use the model's own log-prob as a critic,"
   JEPA's λ should be **0** at this scale. If the downstream goal
   is supervised accuracy with a *separate* verifier, JEPA's
   diversity gain may help — measure on the actual end task.

3. **The negative result is not a JEPA-paper refutation.** The
   published LLM-JEPA settings are 100M+ models, longer training,
   different objectives. We tested λ=0.1, k=8, 5K steps, 50M params.
   A separate-target-encoder or EMA variant, or λ in a different
   regime, could give different numbers. What we falsified is the
   simple claim "JEPA-aux fixes K8's anti-calibration," which was
   the point of running this experiment.

## What's still worth trying (Phase 10 S2 candidates)

- λ sweep at finer granularity (0.01, 0.03, 0.1, 0.3) to find a
  Shape-C–neutral λ that keeps the diversity gain.
- Offset sweep: shorter `k` (1, 2, 4) might be less disruptive.
- EMA target encoder rather than same-encoder stop-gradient.
- Apply JEPA only at later layers, not on the post-`ln_f` output.
- **Different falsifier**: keep K8 and try JEPA on PythonCodeDomain
  where Phase 9 S5 showed pass-rate headroom and the `\n` mode
  isn't dominant — JEPA might help there even if it hurt here.

None of these are committed-to next steps; this entry is the
honest result of the cheapest possible test.

## Reproducing

```bash
# Build (CUDA 12.5)
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p nanogpt-rs --example train_kowiki_jepa \
    --features cuda --release
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p llm-actors --example critic_baseline_korean \
    --features cuda --release

# Train both
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/train_kowiki_jepa \
  --steps 5000 --jepa-lambda 0.0 \
  --save checkpoints/p10s1_baseline.safetensors
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/train_kowiki_jepa \
  --steps 5000 --jepa-lambda 0.1 --jepa-offset 8 \
  --save checkpoints/p10s1_jepa01.safetensors

# Measure
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/critic_baseline_korean \
  --init checkpoints/p10s1_baseline.safetensors
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/critic_baseline_korean \
  --init checkpoints/p10s1_jepa01.safetensors
```

## See also

- `nanogpt-rs/src/jepa.rs` — predictor module, JEPA loss, top-1
  mass metric. 4 unit tests.
- `nanogpt-rs/src/train.rs` — `TrainConfig.jepa_lambda` /
  `jepa_offset`, integrated into `train_from_full`. 2 smoke tests.
- `docs/phase7-design.md` — risk register; risk #12 added based
  on this measurement.
- Phase 9 S2 memory entry — the K8 100K anti-calibration finding
  this experiment was meant to address.
