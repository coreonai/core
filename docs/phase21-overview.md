# Phase 21 — Pekko bridge overview

Phase 17–20 found multi-round SFT + inference-time pass@k on
Qwen2.5-Coder-0.5B and shipped a 4-tier deployment recipe — all
measured via Python scripts. The README's pitch was
"self-evolving agentic foundation model on top of Apache Pekko", but
the actor side spoke only nanogpt_rs::GPT. Phase 21's 10 stages
**migrate the Phase 17–20 recipe (plus an RL extension) onto the
Pekko actor stack** running Candle-native Qwen2 inference and
training. **No Python sidecar required.**

This doc is the **single entry point** to the 10-stage roadmap.
Each stage links to a detailed doc; the overview here gives the
dependency graph, commit hashes, and the run-it-all guide.

## Stages

```
A ─── E ─┬─ E.next ─── E.next.next ─── H ─── G
│        │
C ───────┤
│        │
D ────── ┤
│
F (standalone)
│
B (signal measurement)
```

| stage | scope | commit |
|---|---|---|
| A | Add `passk: usize` to `EvaluatorMessage::Eval`; per-(prompt, k) seed override; `EvaluatorActor` does the loop | `7a5d18b` |
| C | `supervisor::run_multi_round` helper — chained `init_from ← prev save_path`, per-round seed bumps. `RoundConfig` derives `Clone`. | `f09d97d` |
| D | New `qwen_model_actor` module — wraps `candle_transformers::models::qwen2` in a Pekko `Actor<Message = ModelMessage>`. **No Python sidecar.** Two smokes: standalone PoC + actor-pipeline E2E. | `acfdc5d` |
| F | New `qwen2_lora` module — fork of upstream qwen2 with LoRA on `q_proj` + `v_proj` (Phase 14–20 recipe r=16, α=32). `train_qwen_lora_step` standalone helper. **Critical Candle 0.10 gotcha**: `rope`/`softmax_last_dim`/`rms_norm` are `apply_op*_no_bwd` — must use `_slow` variants on the training path or gradients silently die. | `8367d2a` |
| B | Substrate scale-up smoke at n_embd=512 / n_layer=6 (~24M params). pass@k mechanism **replicates** on Rust-native nanogpt: pre-SFT pass@5 +0.333 (1.73×), r=2 SFT pass@5 +0.083 (1.20×). | `8b925a1` |
| E | Generic-ify `EvaluatorActor<M>`, `GeneratorActor<M>`, `RoundActors<M>` over `M: Actor<Message=ModelMessage>` with `default M = ModelActor`. `EvaluatorActor::<QwenModelActor>` is now spellable. | `3ead388` |
| E.next | New `qwen_trainer_actor` module — wraps Stage F's training step in a Pekko Actor with `Train` / `SaveLoraAdapter` / `Ping` messages. | `ec00809` |
| E.next.next | `qwen2_lora::save_merged_lora` helper bakes the LoRA delta into the base safetensors. Output is drop-in for the upstream qwen2 loader so `QwenModelActor::ReloadCheckpoint` picks up training. New `SaveMergedCheckpoint` message. | `8105007` |
| H | `TrainerHandle` trait + `TrainRequest::Sft/Dpo` enum. `RoundActors.trainer: Arc<dyn TrainerHandle>` (was concrete actor ref). Two impls: `TrainerActorHandle` (nanogpt_rs passthrough) + `QwenTrainerActorHandle` (Qwen LoRA). `supervisor::run_round` dispatches via trait — `run_multi_round` now drives **the full Phase 17-20 recipe shape against real Qwen end-to-end**. | `b534679` |
| G | REINFORCE policy-gradient step (`train_qwen_lora_pg_step` + `TrainPolicyGradient` message). Optimizes verifier verdict directly. RLOO baseline-subtracted rewards. | `f8d64ca` |

## Examples (run-it-all guide)

Build the workspace once:

```bash
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
    cargo build --workspace --examples --features cuda --release
```

| stage | example | what it demonstrates |
|---|---|---|
| D | `phase21_qwen_candle_smoke` | Standalone PoC: Candle loads Qwen2.5-Coder-0.5B + generates a valid `def fibonacci(n):` recursive impl. |
| D | `phase21_qwen_actor_smoke` | Actor pipeline E2E: `ActorSystem.spawn(QwenModelActor)` → 3 prompts via `ModelMessage::Generate` → 3 correct Python impls. |
| C | `phase21_multi_round_smoke` | `run_multi_round` orchestration on `ArithmeticDomain` (default `M = ModelActor`). |
| F | `phase21_qwen_lora_smoke` | Standalone training loop: loss 0.8226 → 0.3530 (-57%) over 8 AdamW steps on a fixed corpus. |
| E | `phase21_e_smoke` | `EvaluatorActor::<QwenModelActor>` running pass@1 vs pass@5 — Stage A mechanism flows through Qwen. |
| E.next | `phase21_e_next_smoke` | `QwenTrainerActor` actor-routed Train: same -58% loss as Stage F via `QwenTrainerMessage::Train`. |
| E.next.next | `phase21_e_next_next_smoke` | 3-actor multi-round loop: Train → SaveMergedCheckpoint → ReloadCheckpoint → Eval. Inference picks up training. |
| H | `phase21_h_smoke` | **Full pipeline**: 6 actors + `run_multi_round` drives Gen→Verify→Curate→Train→Reload→Eval against Qwen2.5-Coder-0.5B. |
| G | `phase21_g_smoke` | REINFORCE RL loop: 3 RL steps × 3 prompts × k=2 with RLOO baseline-subtracted rewards. |
| B | `phase21_b_eval_passk` | Clean same-checkpoint pass@k comparison on a `self_improve_rust` checkpoint. Used for the Stage B lift measurement. |

Run them in this order for the storyline:

```bash
./target/release/examples/phase21_qwen_candle_smoke          # Stage D-1
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_qwen_actor_smoke   # Stage D-2
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_multi_round_smoke  # Stage C
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_qwen_lora_smoke    # Stage F
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_e_smoke            # Stage E
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_e_next_smoke       # Stage E.next
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_e_next_next_smoke  # Stage E.next.next
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_h_smoke            # Stage H — full pipeline
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase21_g_smoke            # Stage G — RL
```

Each prints a `phase21_*: PASS` line if everything works. Total
wall-clock for the full sequence on an A100: ~5 minutes (excluding
the one-time CUDA release build).

## Codebase structural changes

New library modules in `llm-actors/src/`:
- `qwen2_lora.rs` — forked Qwen2 with LoRA + training helpers
  (`train_qwen_lora_step`, `train_qwen_lora_pg_step`, `save_merged_lora`)
- `qwen_model_actor.rs` — Pekko Actor wrapping upstream qwen2 for inference
- `qwen_trainer_actor.rs` — Pekko Actor wrapping qwen2_lora for training
- `trainer_handle.rs` — trait abstracting the trainer call,
  with TrainerActorHandle (nanogpt_rs) and QwenTrainerActorHandle (Qwen)

Existing modules generic-ified (Stage E):
- `evaluator_actor.rs` — `EvaluatorActor<M = ModelActor>`
- `generator_actor.rs` — `GeneratorActor<M = ModelActor>`
- `supervisor.rs` — `RoundActors<M = ModelActor>`, `run_round<M>`,
  `run_multi_round<M, F>`

`RoundActors.trainer` (Stage H): `Arc<dyn TrainerHandle>`
(was `ActorRef<TrainerActor>`).

`ModelMessage` gains an optional surface (Stage A):
- `EvaluatorMessage::Eval.passk: usize` and `EvalReport.passk`

## Build / test surface

After Stage G:
- `cargo build --workspace --release` clean
- `cargo build --workspace --examples --release` clean
- `cargo test --workspace --release` — **145 unit tests** pass
  (was 130 pre-Phase-21; net +15 across the 10 stages)
- `cargo fmt --all --check` clean
- `cargo clippy --workspace --all-targets -- -D warnings` clean

## Project-vision status

The README's tagline is "self-evolving agentic foundation model on top
of Apache Pekko". After Phase 21 every step of the Phase 17-20
self-improvement loop against the production Qwen2.5-Coder-0.5B model
runs through Pekko actors with **one function call to
`run_multi_round`**:

| Phase 17-20 step | Phase 21 implementation |
|---|---|
| Eval-before with pass@k | `EvaluatorActor::<QwenModelActor>` (Stage A + E) |
| Generate harvest set | `GeneratorActor::<QwenModelActor>` (Stage E) |
| Verify | `VerifierActor` + `Domain` impl |
| Curate (priority replay) | `CuratorActor` (already existed) |
| Train SFT | `QwenTrainerActor` via `QwenTrainerActorHandle` (E.next + H) |
| Save trained model | `SaveMergedCheckpoint` (E.next.next) |
| Reload | `ModelMessage::ReloadCheckpoint` to `QwenModelActor` (D) |
| Eval-after | same `EvaluatorActor` |
| Multi-round chaining | `supervisor::run_multi_round` (C + H) |

Plus a Stage G RL extension on top.

## What's not in Phase 21 (deferred)

- **On-policy correctness for RL**: Stage G samples from QwenModelActor
  while gradient updates apply to QwenTrainerActor — policy drift over
  many steps. PPO importance weight correction is the textbook fix.
- **`AgenticGeneratorActor<M>` generic-ification**: Phase 4's tool-use
  loop still hardcodes `ActorRef<ModelActor>`. Mechanical change
  (mirror Stage E); not on the Phase 17-20 path.
- **`InferenceServerActor<M>` / `EnsembleActors<M>`**: same situation.
  Phase 5/6 ensemble experiments didn't survive Phase 13 retraction.
- **HumanEval / MBPP as a Rust `Domain`**: Phase 21 examples use
  trivial `PythonReturnDomain` for wiring focus. Real benchmark
  reproduction of Phase 17-20 numbers needs a proper domain impl.
- **Qwen NAS / architecture search**: Phase 3 evolution still operates
  on `nano_50m`. Qwen architecture is fixed at the HF preset.
- **DPO via Qwen**: `QwenTrainerActorHandle::train(Dpo)` bails. Phase
  11/14 settled DPO as non-winning at the LoRA scale, so this is
  intentionally deprioritized.

## Per-stage docs

- `docs/phase21-stage-a.md` — pass@k in `EvaluatorActor`
- `docs/phase21-stage-c.md` — `run_multi_round` helper
- `docs/phase21-stage-d.md` — Candle-native Qwen2 + `QwenModelActor`
- `docs/phase21-stage-f.md` — Candle Qwen2 LoRA training + the no_bwd gotcha
- `docs/phase21-stage-b.md` — substrate scale-up pass@k lift
- `docs/phase21-stage-e.md` — generic Evaluator/Generator/RoundActors
- `docs/phase21-stage-e-next.md` — `QwenTrainerActor`
- `docs/phase21-stage-e-next-next.md` — adapter merge + 3-actor demo
- `docs/phase21-stage-h.md` — `TrainerHandle` + supervisor wiring
- `docs/phase21-stage-g.md` — REINFORCE RL with verifier reward
