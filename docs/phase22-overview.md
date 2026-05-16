# Phase 22 — HumanEval / MBPP on Pekko overview

Phase 21 migrated the Phase 17–20 recipe **shape** onto Pekko —
`supervisor::run_multi_round` driving Candle-native Qwen2.5-Coder-0.5B
through Gen→Verify→Curate→Train→Reload→Eval. But Phase 21's smokes
used a trivial `PythonReturnDomain` ("does the completion contain
`return`?") — wiring-grade, not benchmark-grade. **Phase 22 closes
that gap**: real HumanEval + MBPP substrates as Rust `Domain` impls,
with the Pekko stack now driving actual benchmark measurement,
multi-round SFT, and REINFORCE on top.

This doc is the **single entry point** to the 5-stage roadmap.
Each stage links to a detailed doc; the overview here gives the
dependency graph, commit hashes, measurement results, and the
run-it-all guide.

## Stages

```
A ─── B ─── D ─── D-followup
│          │
C          E
│
mbpp-D variant
```

| stage | scope | commit |
|---|---|---|
| A | `HumanEvalDomain` loads 164 problems, verifies via `python3` subprocess. New `phase22_humaneval_baseline` example. Initial measurement: pass@1 = 0.0793 (greedy) vs Phase 17 S6 ref 0.216 — 2.7× gap, **flagged**. | `91256a4` |
| B | Gap was metric, not bug. Phase 17 S6's "pass@1" = aggregate `total_passes / total_attempts` at temp=0.8 / k=10. New `Domain::n_prompts`/`nth_prompt`, `EvaluatorMessage::EvalSequential { aggregate }`, `EvalReport.{total_attempts, total_passes}`. n=32 × k=10 = 320 attempts: **aggregate pass@1 = 0.222 (71/320)**, matches 0.216 within 1σ (Δ=+0.006, binomial SE=0.023). | `bb78cc3` |
| C | `MbppDomain` mirrors Stage A but for MBPP-100 (task_id 11–110). Phase 17 S3's prompt-synthesis recipe ported to Rust: parse first top-level `def name(args):`, detect imports, emit HumanEval-style prompt + suffix. **Structural diff**: test_list items are top-level asserts, no `check(<entry>)` call. n=16 × k=10 = 160 attempts: aggregate pass@1=0.1875 (subset bias), but Δ pass@1→pass@10 = +0.375 reproduces Phase 17 S9's +0.30 mechanism. | `284000c` |
| D | Multi-round SFT through Pekko. `phase22_he_mr_sft` = Stage 21 H template with `HumanEvalDomain` swap. All Phase 14–20 hyperparams as CLI flags. r=2 smoke: round 0 gen=0/16 (sparse, supervisor honestly `skip training: empty corpus`); round 1 gen=2/16 → trained → pass@3 0.219→0.344 (**Δ=+0.125 lift** — Phase 17 mechanism reproduces even at smoke scale). | `5896d01` |
| E | REINFORCE on HumanEval via Stage G mechanism. `phase22_he_reinforce` = Stage G template with `HumanEvalDomain` + `nth_prompt` indexing. Verifier-as-reward (1.0 if `verify` passes, 0.0 otherwise), RLOO baseline. r=3 smoke (max_new=16) returns clean loss=0.0 in degenerate no-pass case — wiring proven. | `eb6da62` |

### Stage D follow-ups (post-E)

| follow-up | scope | commit |
|---|---|---|
| Anomaly resolution | Round-0 `eval-after=0.000` was a display bug (`unwrap_or(0.0)` conflating `None` with `0.0`). Supervisor early-returns on empty corpus → eval-after never measured. Fixed display to show `N/A`. | `76d0f0e` |
| Sparse-corpus mitigation | `--gen-n` default 16 → 32 (P(skip) 0.185 → 0.034). New `--gen-oversample` CLI flag (Phase 6 Shape C best-of-K filter via `ScoreLogProb`). Doc table for skip probabilities at gen-n ∈ {16, 32, 64, 164}. | `835393f` |
| `--scratch-dir` + `--prompt-skip-list` | Enable parallel `phase22_he_mr_sft` runs (cross-process Mutex isolation). Forward-compat hook for prompt filtering. | `d2d0aa4` |
| `--seed` CLI flag | Make multi-seed runs actually produce distinct results. Internal seeds (gen, eval, corpus) derive deterministically from a single `--seed N` knob. | `1736393` |
| `QwenModelActor::ScoreLogProb` | Was a stub returning `Err`. Implements via the same KV-cache + last-position forward as `generate_autoregressive`. Unlocks `--gen-oversample > 1` on Qwen (Phase 6 Shape C best-of-K filter). | `e787f79` |
| `FilteredDomain` wrapper | Operationalizes `--prompt-skip-list` at the `Domain` trait level (no supervisor/actor changes). Phase 9 S5 cold-start mitigation. 4 new unit tests. | `b9be505` |
| MBPP Stage D variant | `phase22_mbpp_mr_sft` — cross-substrate companion to `phase22_he_mr_sft`. Same actor pipeline, `HumanEvalDomain → MbppDomain` swap. Identical CLI surface. | `2753241` |
| 5-seed gen-n=32 batch | First multi-seed measurement (seeds 100/200/300/400/500). Mean r=2 pass@3 = 0.275 ± 0.116; Δ(r=2−base) = +0.100 ± 0.078 (1.3σ above zero). 5/5 seeds positive. Seed 400 = 0.406 within 0.002 of Phase 17 r=2 = 0.404. σ is 9× Phase 17's due to eval-n=32 + passk=3 subset noise. | `d1dd6d8` |
| 5-seed gen-n=164 A batch | Phase-17-scale gen-n. Mean r=1 = 0.331 (up from gen-n=32's 0.244 — bigger corpus helps single-round SFT). **5/5 seeds r=2 < r=1**: mean Δ(r=2−r=1) = −0.081 — first observation of round-2 regression at this scale (catastrophic forgetting or over-training signal). seed 400 r=1 = 0.562, seed 500 r=1 = 0.438 individually exceed Phase 17's r=2 = 0.404. | TBD |
| `--checkpoint` flag for baseline | Allow `phase22_humaneval_baseline` to evaluate trained checkpoints (overrides `model.safetensors` while reusing snapshot config + tokenizer). Required for benchmark-aligned aggregate eval of Stage D outputs. | TBD |

## Examples (run-it-all guide)

Build once:

```bash
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
    cargo build --workspace --examples --features cuda --release
```

| stage | example | what it demonstrates |
|---|---|---|
| A | `phase22_humaneval_baseline` | Drive `EvaluatorActor::<QwenModelActor>` against `HumanEvalDomain` end-to-end (no standalone driver). Stage A's gap was found here; Stage B's match recipe runs here. |
| B | `phase22_humaneval_baseline --sequential --aggregate` | Reproduces Phase 17 S6's exact metric. `--n-problems 164 --passk 10 --sequential --aggregate` for the full 164×10 = 1640-attempt anchor (~55 min). |
| C | `phase22_mbpp_baseline` | Same shape as A but on MBPP-100. `--sequential --aggregate` reproduces Phase 17 S9. |
| D | `phase22_he_mr_sft` | **Full pipeline**: 6 actors + `run_multi_round` driving Gen→Verify→Curate→Train→Reload→Eval against HumanEval. Every Phase 14–20 hyperparam as a CLI flag. |
| E | `phase22_he_reinforce` | REINFORCE on HumanEval with RLOO-baseline-subtracted rewards. Final Phase 17–20 mechanism with a Rust-native execution path. |

Run them in this order for the storyline:

```bash
# A — initial baseline (greedy on n=32; flagged the metric gap)
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_humaneval_baseline \
    --n-problems 32 --passk 1

# B — aggregate-mode reproduction of Phase 17 S6's 0.216 (~11 min, n=32×k=10):
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_humaneval_baseline \
    --n-problems 32 --passk 10 --sequential --aggregate --max-new-tokens 200

# C — MBPP cross-substrate (~3.5 min, n=16×k=10):
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_mbpp_baseline \
    --n-problems 16 --passk 10 --sequential --aggregate --max-new-tokens 200

# D — MR-SFT through Pekko (~25 min, r=2 smoke):
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_he_mr_sft \
    --rounds 2 --gen-n 16 --eval-n 32 --eval-passk 3 --train-steps 30

# E — REINFORCE (~13 s smoke; bump max-new-tokens for signal):
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_he_reinforce \
    --rl-steps 3 --n-prompts 6 --k-per-prompt 2 --max-new-tokens 16
```

Each prints a `phase22_*: PASS` line if everything works.

## Codebase structural changes

New library modules in `llm-actors/src/`:
- `domain/human_eval.rs` — `HumanEvalDomain`, JSONL loader,
  `verify` via `python3` subprocess with poll-based timeout
- `domain/mbpp.rs` — `MbppDomain`, Phase 17 S3 prompt-synthesis
  port (parse_signature, detect_imports, build_challenge)

`Domain` trait extension (Stage B):
- `n_prompts() -> Option<usize>` — `None` for infinite-prompt
  domains (Arithmetic), `Some(n)` for fixed sets (HumanEval, MBPP)
- `nth_prompt(i) -> Option<String>` — deterministic indexed accessor

`EvaluatorMessage` extension (Stage B):
- New variant `EvalSequential { n, sampling, passk, aggregate, reply }`
  — no-replacement sweep over `domain.nth_prompt(0..n)`
- `EvalReport.{total_attempts, total_passes}` — populated only by
  `EvalSequential { aggregate: true }`, gives Phase 17 S6's
  "pass@1 (raw)" metric directly

## Measurement summary

| stage | substrate | recipe | result | comparison |
|---|---|---|---|---|
| B | HumanEval n=32×k=10 | base Qwen, temp=0.8, BF16 | aggregate pass@1 = **0.222** | matches Phase 17 S6's 0.216 within 1σ (Δ=+0.006) |
| C | MBPP-100 n=16×k=10 | base Qwen, temp=0.8, BF16 | per-prompt pass@10 = 0.5625, aggregate pass@1 = 0.1875, **Δ=+0.375** | mechanism (sampling lift) matches Phase 17 S9's +0.30; absolute level subset-biased (task_id 11–26) |
| D | HumanEval r=2 smoke | gen-n=16, eval-n=32, passk=3, train-steps=30 | round 1 pass@3 0.219→0.344, **Δ=+0.125** | direction matches Phase 17 S1 (mean Δ=+0.174 at full 164 + passk=10); round 0 empty-corpus skip path now displays N/A correctly (the original "0.000" was a display bug, not model state) |
| E | HumanEval r=3 smoke | n_prompts=6, k=2, max_new=16 | 0/12 passes (degenerate), loss=0.0 cleanly | wiring proven; signal-bearing run needs max_new ≥64 + multi-seed |
| D 5-seed gen-n=32 | HumanEval r=2 | 5 seeds, gen-n=32, eval-n=32, passk=3, train-steps=30 | mean r=2 pass@3 = **0.275 ± 0.116**, Δ(r=2−base) = **+0.100 ± 0.078** (1.3σ above zero) | 5/5 seeds positive; seed 400 = 0.406 within 0.002 of Phase 17 r=2 = 0.404; σ 9× Phase 17's due to eval-n + passk noise |
| D 5-seed gen-n=164 (A batch) | HumanEval r=2 | 5 seeds, gen-n=164, eval-n=32, passk=3, train-steps=100 | mean **r=1 pass@3 = 0.331**, mean r=2 = 0.250; **r=2 < r=1 in 5/5 seeds** | NEW finding: at Phase-17-scale gen-n + 100 train-steps, r=2 regresses from r=1 (mean Δ=−0.081). Catastrophic forgetting / over-training signal — not seen at gen-n=32 where r=1 was lower. Phase 17 likely used different train-steps; needs ablation |

## Build / test surface

After Stage E + Stage D follow-ups:
- `cargo build --workspace --release` clean
- `cargo build --workspace --examples --release` clean
- `cargo test --workspace --release` — **160 unit tests** pass
  (was 145 pre-Phase-22; net +15: +4 HumanEval Stage A, +7 MBPP
  Stage C, +4 FilteredDomain)
- `cargo fmt --all --check` clean
- `cargo clippy --workspace --all-targets -- -D warnings` clean

## Project-vision status

After Phase 22, **every Phase 17–20 finding has a Rust-native
execution path**:

| Phase 17–20 finding | Phase 21+22 implementation |
|---|---|
| Base pass@k inference scaling (Phase 17 S6 / S9) | Phase 21 Stage A (`EvaluatorActor.passk`) + Phase 22 Stage B (`EvalSequential.aggregate`) |
| Multi-round SFT saturation curve (Phase 17 S1, 18 S2/S6, 19 S1, 20 S1) | Phase 22 Stage D (`phase22_he_mr_sft`) |
| MR-SFT cross-substrate (Phase 17 SB, 18 S3, 20 S2) | Phase 22 Stage D + Stage C's `MbppDomain` |
| REINFORCE with verifier reward | Phase 22 Stage E (`phase22_he_reinforce`) |
| HumanEval substrate (Phase 14–20) | Phase 22 Stage A+B |
| MBPP substrate (Phase 14–20) | Phase 22 Stage C |

Numerical reproductions of Phase 17's saturation curve (0.230 →
0.404 → 0.475 → 0.519 → 0.556 → 0.581) and Phase 17 S9's MBPP
+0.30 lift are **wallclock exercises on top of this infrastructure**,
not infrastructure gaps. A 5-seed × r=1..6 × 164 × passk=10 full
saturation run is ~165 GPU-h; a 5-seed MBPP r=5 run is ~30 GPU-h.

## What's not in Phase 22 (deferred)

- **Full Phase 17 saturation curve numerical reproduction.** ~165
  GPU-h; binary ready (`phase22_he_mr_sft --rounds 6 --gen-n 164
  --eval-n 164 --eval-passk 10 --train-steps 100`). The original
  "round-0 anomaly" turned out to be a display bug
  (`unwrap_or(0.0)` conflated "skipped" with "measured zero"); the
  supervisor's `skip training: empty corpus` early-return cleanly
  no-ops save/reload/eval-after and leaves the model unchanged.
  The real prep work is sparse-corpus robustness: at p≈0.10
  per-attempt, P(empty 0/16 corpus) ≈ 0.185. Stage D's
  `phase22_he_mr_sft --gen-n` default is now 32 (P(skip)≈3% per
  round); for a clean 6-round saturation sweep use `--gen-n 64` or
  `--gen-n 164`. `--gen-oversample` is a separate quality lever
  (best-of-K by `ScoreLogProb`), not a quantity multiplier — see
  Stage D doc for the distinction.
- **Aggregate eval inside supervisor.** Per-round eval uses
  `EvalRandom`; benchmark-aligned aggregate is a separate
  `phase22_humaneval_baseline` invocation on the saved
  `r{R}_merged.safetensors`. Worth wiring `RoundConfig` to support
  `EvalSequential { aggregate }`, separate scope.
- **Adapter-sync between RL steps (Stage E).** QwenModelActor
  weights drift from QwenTrainerActor's LoRA delta during RL;
  `SaveMergedCheckpoint + ReloadCheckpoint` between RL steps would
  re-sync. Deferred for memory budget reasons.
- **Importance-weighting correction (Stage E).** Off-policy
  theoretical correctness; not required for the smoke to demonstrate
  the loop. Same situation as Phase 21 Stage G.
- **Pre-filtered prompt set for RL.** Phase 9 S5 cold-start
  observation argues for filtering out problems the base model has
  ≥k failures on (their RLOO baseline always = 0, no gradient).
  Trade-off with selection bias; Stage E ships the unfiltered version.
- **Concurrent verify.** Both `HumanEvalDomain::verify` and
  `MbppDomain::verify` serialize python3 subprocess calls under a
  `write_lock`. 164×10 = 1640 serial verifies is the wallclock
  bottleneck. A multi-process executor (one scratch dir per worker)
  is straightforward but separate scope.

## Per-stage docs

- `docs/phase22-stage-a.md` — `HumanEvalDomain` + baseline binary
- `docs/phase22-stage-b.md` — `EvalSequential.aggregate` reproduces
  Phase 17 S6's 0.216
- `docs/phase22-stage-c.md` — `MbppDomain` cross-substrate mirror
- `docs/phase22-stage-d.md` — MR SFT through Pekko on HumanEval
- `docs/phase22-stage-e.md` — REINFORCE on HumanEval

## See also

- `docs/phase21-overview.md` — the Pekko bridge this builds on
- `scripts/phase17_s6/run_passk.py` — Phase 17 S6's reference HumanEval pass@k script
- `scripts/phase17_s9/run_passk_mbpp.py` — Phase 17 S9's reference MBPP script
- `scripts/phase17_s3/problems.py` — Phase 17 S3's MBPP prompt-synthesis recipe (port basis for Stage C)
