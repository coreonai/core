# Phase 22 — Final consolidation (Stage A → ROOT-CAUSE FIX → recipe match)

End-of-Phase-22 narrative summary. Captures the full arc from
substrate setup (Stage A-E) through the catastrophic regression
diagnosis (A-batch → G1 → G2) to the root-cause discovery (Phase 17
recipe byte-comparison) and recipe match (G4 mask-only, G5 mask+cosine).

## Phase 22's purpose

Migrate Phase 17-20's HumanEval/MBPP recipe from Python (HF
Transformers + PEFT) onto Pekko-Rust (Candle-native Qwen2 + LoRA).
The benchmark target: reproduce Phase 17 S1's r=2 HumanEval pass@1 =
0.404 ± 0.013 through `supervisor::run_multi_round` on actor stack.

## Stage matrix

| Stage | Scope | Commit | Status |
|---|---|---|---|
| A | HumanEvalDomain (164 problems) | `91256a4` | ✅ |
| B | EvalSequential + aggregate (Phase 17 metric reproduces ≈ within 1σ) | `bb78cc3` | ✅ |
| C | MbppDomain (cross-substrate) | `284000c` | ✅ |
| D | Multi-round SFT through Pekko | `5896d01` | ✅ wiring shipped, regression in measurement |
| E | REINFORCE on HumanEval | `eb6da62` | ✅ wiring shipped |

## The Stage D regression saga

| Batch | Recipe | Aggregate r=2 pass@1 | Diagnosis |
|---|---|---|---|
| A-batch | train-steps=100, gen-n=164 | 0.116 ± 0.037 | HALF base 0.222 (catastrophic) |
| G1 | train-steps=30 | 0.154 ± 0.032 | partial recovery, still below base |
| G2 MBPP | train-steps=30 | (per-round) 0.244 < base 0.313 | substrate-divergent damage |
| G4 | train-steps=30 + **mask-prompt** | 0.117 ± 0.048 | mask alone insufficient (≈ A-batch) |
| G5 | train-steps=30 + mask + **cosine LR** | 0.152 ± 0.063 | recipe-detail ports net-zero vs G1 |
| G6 | samples=6 systematic + batch=1 + ts=200 | 0.108 (1-seed) | data volume FALSIFIED; isolation: single round craters base 0.222→0.085 = step-level batch=1 overfitting |
| G7 | + **batch=4** padded SFT | 0.173 ± 0.009 (4-seed) | collapse FIXED (σ 0.063→0.009, loss blowup gone) but still < base |
| G8 | + **fresh AdamW/round** + **non-cumulative buffer** | 0.2037 (1-seed) | round-level flags ≈ no effect (G8 seed800 0.204 ≈ G7 seed700 0.209); loop lands ≈base, gap to 0.404 unclosed |

## Infrastructure shipped this session

### Recipe fixes (close the Phase 17 gap)

| Component | Commit |
|---|---|
| Prompt-mask completion-only loss (`train_qwen_lora_step_masked` + `cross_entropy_with_prompt_mask` + `TrainSftPairs` actor message + `RenderPairs` curator message + `TrainRequest::Sft.sft_pairs` + `RoundConfig.sft_mask_prompt`) | `bc90db5`, `3405173` |
| Cosine LR schedule with 10% warmup | `c7a7aed` |
| `batch_size > 1` + padding | (deferred — wallclock opt, not correctness) |

### Stage D follow-up infrastructure (independent wins)

| Component | Commit |
|---|---|
| Display fix for empty-corpus round (anomaly resolved) | `76d0f0e` |
| Sparse-corpus mitigation (`--gen-n` default 16→32) | `835393f` |
| `--scratch-dir` + `--prompt-skip-list` | `d2d0aa4` |
| `--seed` CLI flag (multi-seed RNG fan-out) | `1736393` |
| FilteredDomain wrapper + 4 unit tests | `b9be505` |
| MBPP variant binary `phase22_mbpp_mr_sft` | `2753241` |
| `--checkpoint` flag for `phase22_humaneval_baseline` | `d861a36` |
| QwenModelActor::ScoreLogProb implementation | `e787f79` |
| QwenModelActor::LossOn implementation (C1) | `7d2c7c0` |
| REINFORCE `--sync-every` adapter sync (C2) | `905adfd` |
| tmpfs hint on `--out-dir` (C3) | `7d2c7c0` |
| Thread-safe Domain.verify + VerifierActor parallel pool (#3) | `38025bd` |
| `run_multi_round` eval-before dedup (#1) | `cb6a9b7` |
| EvaluatorActor aggregate-mode pipelined verify (#2) | `d9834a6` |
| CPU fail-fast guard + CLAUDE.md gotcha #8 | `e030fce` |
| CLAUDE.md gotcha #9 (recipe byte-compare) + E bundle | `10799cb` |

### Tests: 145 (pre-Phase-22) → **167** (post-fix)

- +4 HumanEval Stage A
- +7 MBPP Stage C
- +4 FilteredDomain
- +1 parallel-verify regression
- +4 prompt-mask helper
- +2 cosine LR schedule

## Final measurement table (G4 → G9)

The G5-G8 aggregates were measured **without** eval-truncation (the
pre-G9 baseline binary). G9 added completion truncation at both train
and eval; the G9 rows + base-trunc are measured **with** eval-truncation
(matching Phase 17). So the honest comparison is the paired
**base-trunc vs G9** at the bottom — both use the same metric Phase 17
used.

| Configuration | Mean r=2 pass@1 | σ | n | eval-trunc | Note |
|---|---|---|---|---|---|
| Base Qwen (Stage B) | 0.222 | — | — | no | original anchor |
| A-batch (pre-fix) | 0.116 | 0.037 | 5 | no | catastrophic regression |
| G4 (mask-only) | 0.117 | 0.048 | 5 | no | mask alone insufficient |
| G5 (mask + cosine) | 0.152 | 0.063 | 5 | no | net-zero vs G1 |
| G6 (samples=6, batch=1, ts=200) | 0.108 | — | 1 | no | data volume falsified |
| G7 (+ batch=4) | 0.170 | 0.011 | 5 | no | collapse fixed, σ 7× tighter, still < base |
| G8 (+ fresh-opt + non-cumul) | 0.204 | — | 1 | no | round-level flags ~no effect |
| **base (truncated)** | **0.218** | — | — | yes | matches Phase 17 base 0.216 |
| **G9 (+ truncation) 1-seed** | **0.438** | — | 1 | yes | matches/exceeds Phase 17 |
| **G9 (+ truncation) 5-seed** | **0.4356** | **0.016** | 5 | yes | **reproduces Phase 17** |
| Phase 17 reference (S1) | 0.404 | 0.013 | 5 | yes | (target) |

**Conclusion (Phase 22 Stage D — RESOLVED):** the full Phase-17 recipe
is `{completion-mask, cosine LR, batch=4, fresh AdamW/round,
non-cumulative harvest, **completion truncation**}`. The −0.20 gap that
survived G4-G8 was the missing truncation: `build_program` fed the RAW
model completion, and neither generator nor evaluator truncated it
(`stop_char = None` for multi-line HumanEval). Raw completions carried
trailing test scaffolding (`def test_`, `print(`, `if __name__`), leaked
`<|fim_middle|>` tokens, and cut-off-mid-statement tails — which (a)
failed the verifier (harvest yield ~80 instead of ~200 pairs/round) and
(b) failed at eval. Porting Phase 17's `truncate_completion`
(`Domain::truncate_completion` + `truncate_python_completion`, commit
`aaf0594`) fixed both:

- **base (truncated) = 0.218** ≈ Phase 17 base **0.216**
- **G9 r=2 (5-seed) = 0.4356 ± 0.016** ≈ Phase 17 r=2 **0.404 ± 0.013**

The +0.218 lift (0.218 → 0.436, a clean doubling) tracks Phase 17's
+0.188. **The Rust/Pekko self-evolving MR-SFT loop now reproduces
(slightly exceeds) the Phase 17 Python reference — the full
self-evolving loop is end-to-end native in Rust + Pekko.** (The LoRA
merge math was byte-compared and proven consistent — it was NOT the
bug; the divergence was data preprocessing on both sides of the
training loop.)

### Recipe resolution order (the −0.20 gap was a stack)

mask (`bc90db5`) → cosine LR (`c7a7aed`) → batch=4 (`59aab8d`) →
fresh-opt + non-cumul (`e69de7e`, ~neutral on their own) →
**truncation (`aaf0594`, decisive)**.

### Saturation curve (G9 recipe, `--rounds 3`, truncated aggregate)

| round | pass@1 | Δ vs prev | n | Phase 17 ref |
|---|---|---|---|---|
| base | 0.218 | — | — | 0.216 |
| r=2 | 0.436 ± 0.016 | +0.218 | 5 | 0.404 |
| r=3 | 0.4645 ± 0.026 | +0.029 | 5 | 0.475 |
| r=4 | (in flight) | — | — | 0.519 |

(seed 400 OOM'd at round-2 training 3× on contended shared GPUs —
infra, not recipe — and completed on a clean GPU for the 5th seed.)

The r1→r2 jump (+0.218) decelerates to +0.029 at r2→r3 — **diminishing
returns / plateau onset**, the same shape as Phase 17 (r2→r3 +0.071).
r=3 0.465 ≈ Phase 17 r=3 0.475 within noise: the Rust/Pekko loop
reproduces not just the r=2 point but the **saturation curve shape**.

## Recipe recommendation (final)

After G4 + G5 land. Skeleton:

```bash
# Recommended Phase 22 Stage D recipe for production:
cargo run -p llm-actors --example phase22_he_mr_sft \
    --features cuda --release -- \
    --seed N \
    --rounds 2 \
    --gen-n 164 --eval-n 32 --eval-passk 3 \
    --train-steps 30 \
    --max-new-tokens 200 \
    --out-dir /dev/shm/phase22_he_mr_sft_seed_N \
    --scratch-dir /dev/shm/phase22_he_scratch_seed_N
```

Defaults guarantee:
- `sft_mask_prompt = true` (RoundConfig default) → completion-only CE
- Cosine LR schedule with 10% warmup → applied in TrainSftPairs path
- VerifierActor parallel pool (8 workers) → faster batch verify
- Aggregate-mode verify pipelining in EvaluatorActor → faster aggregate eval
- `--out-dir /dev/shm/...` → tmpfs cuts ~10-15s/round disk I/O

## Open items (Phase 23 candidates)

- D7: batch_size > 1 with padding (Phase 17 used batch=4). Wallclock
  optimization; not correctness-critical now that mask + cosine land.
- Adapter-sync between RL steps (Stage E C2 shipped the flag,
  measurement work remains).
- LoRA rank-32 ablation at the fixed recipe.
- Full Phase 17 saturation curve (r=1..6 × 5 seeds × full 164 ×
  passk=10) — true numerical reproduction at scale.
- Pre-filtered prompt set via FilteredDomain (Phase 9 S5 cold-start
  mitigation).

## See also

- `docs/phase22-overview.md` — Phase 22 single entry point
- `docs/phase22-stage-d-A-batch-gen-n-164.md` — A-batch result
- `docs/phase22-stage-d-train-steps-ablation.md` — G1 result
- `docs/phase22-stage-d-G2-mbpp-cross-substrate.md` — G2 result
- `memory/phase22_stage_d_root_cause_fix.md` — discovery writeup
- Notion: workLLM — Phase 22 Stage D ROOT-CAUSE 발견 (Phase 17 복구)
