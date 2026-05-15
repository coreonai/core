# Phase 21 Stage A — Pekko actor pass@k integration

Phase 20 closed with multi-round SFT saturation curve (HE r=5 plateau,
MBPP r=5 cross-substrate) and a 4-tier deployment recipe — all
measured via standalone Python scripts against Qwen2.5-Coder-0.5B.

Phase 21 closes the gap to the project vision (**self-evolving
agentic foundation model on top of Apache Pekko**) by wiring the
Phase 17 S6 / S8 / S9 inference-time pass@k mechanism into the
`llm-actors` Rust actor stack.

## What's in this commit

### Pass@k at the eval phase

`EvaluatorMessage::Eval` gains a `passk: usize` field. The actor's
`run()` loops k times per prompt with **deterministic per-(prompt, k)
seed overrides** so the eval remains fully reproducible, and counts a
prompt correct if any of the k completions verifies.

Short-circuits: once a prompt verifies it stops sampling (cheaper on
easy prompts).

`EvalReport` gains a `passk` field echoed back from input, so
trajectories and logs carry the eval shape.

### RoundConfig wiring

`supervisor::RoundConfig` gains `eval_passk: usize`. `1` (default)
preserves the historical pass@1 behavior. The supervisor's
`ask_eval` helper now plumbs this through to both eval-before and
eval-after phases.

### Driver wiring

- `self_improve_round` example: hardcoded `eval_passk: 1` (arithmetic
  task is greedy by design).
- `self_improve_korean` example: hardcoded `eval_passk: 1` (Korean
  greedy decode pattern from Phase 1).
- `self_improve_rust` example: new `--eval-passk <k>` CLI flag, default
  `1`. Phase 21+ K9 / Rust experiments can opt in to k > 1 to surface
  stochastic-decode capability.
- `self_improve_ensemble_rust` example: hardcoded `passk: 1` on the
  local helper (ensemble is its own consensus mechanism).

## Acceptance criteria — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **136 tests** (83 + 53), all pass
- ✅ `cargo fmt --all --check`: clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings`: clean
- ✅ E2E smoke: `self_improve_rust --rounds 2 --pretrain-steps 800
  --round-train-steps 200 --eval-passk 5` runs end-to-end with
  `EvaluatorActor done passk=5` in the log — pipeline integrated.

## Why no signal measurement at K9 scale (yet)

E2E smoke ran cleanly but eval-after stayed at 0/24 in both pre-train
and post-train — exactly the K9 noise floor Phase 13 retracted as
unsuitable for algorithmic comparison (σ_within > 0.14 of the mean).
At the 1M-param tiny char-model scale, the model can't get **any**
correct verifier slot with the smoke-sized pretrain, so pass@k has
nothing to "rescue" — the lift mechanism (broaden the chosen
distribution) needs a baseline > 0.

This matches Phase 17 S6's mechanism story: pass@k buys signal where
greedy is *near but not at* the answer. K9 1M is far from the answer.

### Where pass@k WILL surface lift at this stack

Two paths, both Phase 21 Stage B+ candidates:

1. **Substrate scale-up via Pekko**: same `self_improve_rust` driver
   but with `--n-embd 512 --n-layer 6 --n-head 8` (Phase 13 S3b
   Stage B) and longer pretrain. ~5-10× the K9 cost, surfaces the
   greedy / sampling gap.
2. **Qwen-via-Pekko integration**: wrap the HF Qwen model behind a
   `ModelActor` impl, then run Phase 17-20 recipes with pass@k at
   the Rust orchestration layer. Significant infra work — the
   `ModelActor` currently owns a `VarMap` + Candle `GPT`, not an
   external Python model. A wrapper actor that shells out (or
   talks to a local HF inference server) would close the gap.

Path 2 directly bridges Phase 17-20 Python findings into the actor
framework that the README sells. Path 1 is the **honest in-stack
measurement** (no Python dependency).

## What this commit does NOT do

- No new measurement at substrate scale. Path 1 above is Stage B.
- No `ModelActor` impl for external HF/transformers models (Path 2).
- No multi-round helper API. Examples already loop over `run_round`;
  a `run_multi_round(actors, configs)` wrapper would be nice but
  not load-bearing.
- No pass@k for the **generate** phase (Phase 6 Shape C oversample
  already covers that axis differently — generator-side).

## Phase 21 Stage roadmap (post Stage A)

| stage | scope | gating |
|---|---|---|
| **A** (this commit) | Pass@k infra in actor stack | ✅ |
| B | Substrate scale-up (n_embd=512, n_layer=6) + measure passk lift | needs ~5-10× K9 wallclock |
| C | Multi-round helper `run_multi_round` + smoke at scale-up | follow B |
| D | HF Qwen `ModelActor` impl (Pekko ↔ Python bridge) | days of infra work |
| E | RL with pass@k reward (Phase 19 candidate #4) | days of code |

## Files

- `llm-actors/src/evaluator_actor.rs` — `passk` field + per-k seed loop
- `llm-actors/src/supervisor.rs` — `RoundConfig.eval_passk` + ask_eval plumbing
- `llm-actors/examples/self_improve_rust.rs` — `--eval-passk` CLI flag
- `llm-actors/examples/self_improve_round.rs` — `eval_passk: 1`
- `llm-actors/examples/self_improve_korean.rs` — `eval_passk: 1`
- `llm-actors/examples/self_improve_ensemble_rust.rs` — `passk: 1`
- `docs/phase21-stage-a.md` (this)

## See also

- `docs/phase20-closeout.md` — saturation curve + deployment recipe
- `docs/phase17-closeout.md` — pass@k discovery (Python side)
- Notion: workLLM — Phase 20 종결
