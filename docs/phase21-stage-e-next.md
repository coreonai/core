# Phase 21 Stage E.next — `QwenTrainerActor`

Stage F shipped `train_qwen_lora_step` as a standalone function (no
actor wrapping). Stage D shipped `QwenModelActor` for inference. Stage
E.next wraps the training side in a Pekko actor so the full Phase
14-20 production model — Qwen2.5-Coder-0.5B — runs **both inference
AND training** through the actor framework. No Python sidecar.

## What's in this commit

### `llm-actors/src/qwen_trainer_actor.rs`

`QwenTrainerActor` (own type) with messages:

| message | semantics |
|---|---|
| `Train { texts, train_steps, reply }` | tokenize once, then run `train_qwen_lora_step` `train_steps` times round-robin over `texts`. Reports per-step loss trajectory + initial/final loss. |
| `SaveLoraAdapter { path, reply }` | `VarMap::save(path)` — persists ONLY the LoRA adapter Vars to a safetensors file. Frozen base is left untouched. |
| `Ping { reply }` | trivial health check |

Constructor `from_snapshot_dir(snapshot, device, dtype, lora_cfg, lr)`
loads `config.json` + `tokenizer.json` + `model.safetensors` from one
HF cache snapshot, attaches fresh LoRA adapters (Phase 14-20 recipe
`r=16, α=32` by default), and builds AdamW over only the LoRA Vars.
Frozen base mmaps as the upstream `qwen2` module does.

### `phase21_e_next_smoke` (CUDA E2E)

```
[Phase21E.next] QwenTrainerActor built  lora_params = 1081344
[Phase21E.next] Ping OK
[Phase21E.next] sending Train { texts: 3 examples, train_steps: 8 }
[Phase21E.next] loss trajectory (8 steps):
  step 0  loss = 0.8226
  step 1  loss = 0.6377
  step 2  loss = 1.1198
  step 3  loss = 0.6215
  step 4  loss = 0.4693
  step 5  loss = 0.7947
  step 6  loss = 0.5319
  step 7  loss = 0.3491
[Phase21E.next] initial=0.8226  final=0.3491  Δ=-0.4735
[Phase21E.next] SaveLoraAdapter OK  path=...  size=4336336 bytes
phase21_e_next_smoke: PASS
```

Per-step variance comes from cycling through 3 different texts —
each step's loss is measured against a different prompt. The
**initial → final delta (-58%)** matches Stage F's single-text smoke
(0.8226 → 0.3530, -57%), so the actor wrapping adds no overhead
beyond the message dispatch.

Adapter file size = 4.3 MB. The base Qwen2.5-Coder-0.5B safetensors is
~1.5 GB — only LoRA Vars are persisted, base stays mmapped.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **145 tests** (was 144 in
  Stage E; +1 compile-time `Actor<Message=QwenTrainerMessage>`
  assertion in `qwen_trainer_actor::tests`)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_e_next_smoke` PASS — loss 0.8226 → 0.3491 over 8
  AdamW steps via `QwenTrainerMessage::Train`; adapter persisted
  via `SaveLoraAdapter`

## What this does NOT do (deferred)

- **`run_multi_round` integration**. `RoundActors.trainer:
  ActorRef<TrainerActor>` is hardcoded to the nanogpt_rs trainer.
  Plumbing `QwenTrainerActor` in needs either:
  - `RoundActors<M, T>` generic over the trainer too, OR
  - A trait that abstracts "the training step" so `TrainerActor` and
    `QwenTrainerActor` both implement it
  Both are bigger surgery; this commit ships the actor itself
  (Stage F mechanism in Pekko clothing), and a focused E.next.next
  can wire it into multi-round.

- **Merged-base safetensors export**. After training, callers get a
  LoRA-only adapter file. To hand that off to `QwenModelActor` (which
  uses upstream `qwen2` — no LoRA hooks), the adapter has to be
  merged back into the base: `W' = W + (B @ A) * (α / r)` per LoRA
  layer. Short follow-on; not in this commit.

- **`LoadLoraAdapter`**. `SaveLoraAdapter` round-trips with a future
  `LoadLoraAdapter` for incremental training. Punted — `VarMap::load`
  is the underlying call.

## Pekko bridge status — both halves now Rust-native

After Stage E.next, the **inference + training** halves of the Pekko
vision against the Phase 14-20 production Qwen model are both
end-to-end Rust:

| half | actor | mechanism |
|---|---|---|
| inference | `QwenModelActor` (Stage D) | `candle_transformers::models::qwen2` |
| training  | `QwenTrainerActor` (this) | `qwen2_lora::train_qwen_lora_step` |

Only `run_multi_round` orchestration remains as a "glue" gap —
both individual halves are first-class Pekko actors and can be
driven from any tokio task or actor system.

## Phase 21 stage roadmap (post Stage E.next)

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 + QwenModelActor (inference) | ✅ (`acfdc5d`) |
| F | Candle-native Qwen2 LoRA training (standalone) | ✅ (`8367d2a`) |
| B | Substrate scale-up + pass@k lift | ✅ (`8b925a1`) |
| E | Generic Evaluator/Generator over `Actor<Message=ModelMessage>` | ✅ (`3ead388`) |
| **E.next** | **`QwenTrainerActor` (Trainer side of Pekko bridge)** | ✅ (this commit) |
| E.next.next | RoundActors generic over trainer type → full P17-20 recipe via Pekko | next-up |
| G | RL with pass@k reward | deferred |

## Files

- `llm-actors/src/qwen_trainer_actor.rs` — new actor + 1 unit test
- `llm-actors/src/lib.rs` — re-export
- `llm-actors/examples/phase21_e_next_smoke.rs` — new E2E demo
- `llm-actors/Cargo.toml` — example registration
- `docs/phase21-stage-e-next.md` (this)

## See also

- `docs/phase21-stage-f.md` — standalone `train_qwen_lora_step` + Candle 0.10 no_bwd gotcha
- `docs/phase21-stage-e.md` — generic `EvaluatorActor`/`GeneratorActor`/`RoundActors`
- `docs/phase21-stage-d.md` — `QwenModelActor` inference
