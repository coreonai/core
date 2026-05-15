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
A ─── B ─── D
│          │
C          E
```

| stage | scope | commit |
|---|---|---|
| A | `HumanEvalDomain` loads 164 problems, verifies via `python3` subprocess. New `phase22_humaneval_baseline` example. Initial measurement: pass@1 = 0.0793 (greedy) vs Phase 17 S6 ref 0.216 — 2.7× gap, **flagged**. | `91256a4` |
| B | Gap was metric, not bug. Phase 17 S6's "pass@1" = aggregate `total_passes / total_attempts` at temp=0.8 / k=10. New `Domain::n_prompts`/`nth_prompt`, `EvaluatorMessage::EvalSequential { aggregate }`, `EvalReport.{total_attempts, total_passes}`. n=32 × k=10 = 320 attempts: **aggregate pass@1 = 0.222 (71/320)**, matches 0.216 within 1σ (Δ=+0.006, binomial SE=0.023). | `bb78cc3` |
| C | `MbppDomain` mirrors Stage A but for MBPP-100 (task_id 11–110). Phase 17 S3's prompt-synthesis recipe ported to Rust: parse first top-level `def name(args):`, detect imports, emit HumanEval-style prompt + suffix. **Structural diff**: test_list items are top-level asserts, no `check(<entry>)` call. n=16 × k=10 = 160 attempts: aggregate pass@1=0.1875 (subset bias), but Δ pass@1→pass@10 = +0.375 reproduces Phase 17 S9's +0.30 mechanism. | `284000c` |
| D | Multi-round SFT through Pekko. `phase22_he_mr_sft` = Stage 21 H template with `HumanEvalDomain` swap. All Phase 14–20 hyperparams as CLI flags. r=2 smoke: round 0 gen=0/16 (sparse, supervisor honestly `skip training: empty corpus`); round 1 gen=2/16 → trained → pass@3 0.219→0.344 (**Δ=+0.125 lift** — Phase 17 mechanism reproduces even at smoke scale). | `5896d01` |
| E | REINFORCE on HumanEval via Stage G mechanism. `phase22_he_reinforce` = Stage G template with `HumanEvalDomain` + `nth_prompt` indexing. Verifier-as-reward (1.0 if `verify` passes, 0.0 otherwise), RLOO baseline. r=3 smoke (max_new=16) returns clean loss=0.0 in degenerate no-pass case — wiring proven. | `eb6da62` |

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
| D | HumanEval r=2 smoke | gen-n=16, eval-n=32, passk=3, train-steps=30 | round 1 pass@3 0.219→0.344, **Δ=+0.125** | direction matches Phase 17 S1 (mean Δ=+0.174 at full 164 + passk=10); round 0 empty-corpus anomaly flagged |
| E | HumanEval r=3 smoke | n_prompts=6, k=2, max_new=16 | 0/12 passes (degenerate), loss=0.0 cleanly | wiring proven; signal-bearing run needs max_new ≥64 + multi-seed |

## Build / test surface

After Stage E:
- `cargo build --workspace --release` clean
- `cargo build --workspace --examples --release` clean
- `cargo test --workspace --release` — **156 unit tests** pass
  (was 145 pre-Phase-22; net +11: +4 HumanEval Stage A, +7 MBPP
  Stage C)
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
  --eval-n 164 --eval-passk 10 --train-steps 100`). Gated on the
  Stage D round-0 empty-corpus anomaly being resolved first.
- **Stage D anomaly investigation.** When the supervisor `skip
  training: empty corpus`s on round 0, eval-after still drops to
  0.000 from 0.219 — implies save/reload cycle perturbs model state
  even with no LoRA delta change. Possible causes: LoRA init isn't
  exactly zero-delta in F32, `SaveMergedCheckpoint` dtype quirk,
  reload-side state mutation. Round 1's positive +0.125 lift
  proceeds normally, so the issue is localized to the empty-corpus
  path.
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
