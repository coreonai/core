# Phase 21 Stage H — supervisor pipeline drives QwenTrainerActor

Stage E.next.next demonstrated the Pekko bridge in a hand-rolled
multi-round loop. The supervisor's `run_round` / `run_multi_round`
still hardcoded `ActorRef<TrainerActor>` (nanogpt_rs trainer), so
QwenTrainerActor couldn't ride the official orchestration.

Stage H introduces the `TrainerHandle` trait, makes
`RoundActors.trainer: Arc<dyn TrainerHandle>`, and provides two
concrete handles: `TrainerActorHandle` (existing nanogpt_rs flow) and
`QwenTrainerActorHandle` (Stage E.next Qwen path). **`run_multi_round`
now drives the full `Gen → Verify → Curate → Train → Reload → Eval`
cycle against the real Qwen2.5-Coder-0.5B production model**.

## What's in this commit

### `llm-actors/src/trainer_handle.rs` (new)

```rust
pub enum TrainRequest {
    Sft { corpus, save_path, init_from, train_cfg, anchor, freeze_base },
    Dpo { pairs, save_path, init_from, reference_path, train_cfg, beta, sft_anchor_weight },
}

#[async_trait]
pub trait TrainerHandle: Send + Sync + 'static {
    async fn train(&self, req: TrainRequest) -> anyhow::Result<TrainOutcome>;
}
```

Two concrete implementations:
- `TrainerActorHandle` — wraps `ActorRef<TrainerActor>` and dispatches
  Sft → `TrainerMessage::Train`, Dpo → `TrainerMessage::TrainDpo`.
  Matches pre-Stage-H behavior exactly.
- `QwenTrainerActorHandle` — wraps `ActorRef<QwenTrainerActor>` +
  `train_steps` + `base_safetensors` path. On Sft: split corpus by
  newlines → texts, send `QwenTrainerMessage::Train`, then
  `SaveMergedCheckpoint { base_path, out_path: save_path }`. On Dpo:
  bails (Qwen LoRA path is SFT-only).

### `llm-actors/src/supervisor.rs`

- `RoundActors<M>.trainer: Arc<dyn TrainerHandle>` (was
  `ActorRef<TrainerActor>`)
- `run_round<M>` builds the corpus or pair set via the curator, then
  hands a `TrainRequest` to `actors.trainer.train(req).await`. The
  DPO vs SFT branching lives at the supervisor level; the actor type
  is abstracted away.

### Existing examples updated

`self_improve_round`, `self_improve_korean`, `self_improve_rust`,
`phase21_multi_round_smoke` all wrap their `ActorRef<TrainerActor>` in
`Arc::new(TrainerActorHandle::new(trainer_ref))` before constructing
`RoundActors`. Zero semantic change — TrainerActorHandle just
forwards to the existing TrainerMessage variants.

### `llm-actors/examples/phase21_h_smoke.rs` (new)

Full Pekko pipeline against Qwen2.5-Coder-0.5B:

```
QwenModelActor (F16 inference)
    ↑
GeneratorActor::<QwenModelActor>  ←  Gen phase
VerifierActor (PythonReturnDomain)  ←  Verify phase
CuratorActor                        ←  Curate phase
QwenTrainerActor (F32 training)
    via QwenTrainerActorHandle      ←  Train + SaveMergedCheckpoint
QwenModelActor::ReloadCheckpoint     ←  Reload
EvaluatorActor::<QwenModelActor>    ←  Eval-after
```

Output (2 rounds):
```
[Phase21H] 6 actors spawned + RoundActors built
phase: generate    round=1  generated=4/4  errors=0
phase: verify      round=1  verified=4 correct=4
phase: curate      round=1  accepted=2 buffer=4
phase: train (SFT) round=1  QwenTrainerActor train done  steps=4
phase: reload      round=1  QwenModelActor checkpoint reloaded
phase: eval-after  round=1  total=6 correct=6 passk=1 pass_rate=1.0

[Phase21H] round 1  gen=4/4  eval_before=6/6  eval_after=6/6

supervisor::run_multi_round drove the full
Gen → Verify → Curate → Train → Reload → Eval cycle
against Qwen2.5-Coder-0.5B end-to-end.

phase21_h_smoke: PASS
```

Pass-rate stays at 6/6 because the smoke uses the trivial
`PythonReturnDomain` (completion contains `"return"`) — Qwen always
produces a function body with `return`. **The Stage H deliverable is
the supervisor pipeline driving real Qwen end-to-end**, not a
benchmark signal.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **145 tests** (no change)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_h_smoke` PASS — 2 rounds of full Gen-Verify-Curate-
  Train-Reload-Eval against Qwen2.5-Coder-0.5B via
  `supervisor::run_multi_round`
- ✅ Existing examples (`self_improve_round` / `_rust` / `_korean` /
  `_ensemble_rust` / `phase21_multi_round_smoke`) continue to compile
  + work via `TrainerActorHandle` wrapping (no behavioral change)

## Project-vision status — the README pitch is realized

The README's tagline is "self-evolving agentic foundation model on top
of Apache Pekko". After Stage H, **every step of the self-improvement
loop** against the Phase 14-20 production model runs through Pekko
actors with a single function call to `run_multi_round`:

| README phase | implementation |
|---|---|
| 1 (model) | `QwenModelActor` via `candle_transformers::qwen2` |
| 2 (improve loop) | `supervisor::run_multi_round` ↔ 6 actors |
| 3 (NAS) | Phase 3 evolution still on nano_50m; Qwen is fixed |
| 4 (tool use) | AgenticGeneratorActor not generic-ified yet |

The remaining gaps are NAS over Qwen architectures (rarely useful —
HF preset is the practical choice) and tool-use orchestration with
Qwen. Both are orthogonal extensions; the **core self-improvement
infrastructure on Pekko is complete**.

## What's deferred

- **Multi-seed benchmark via Pekko**: needs a `--seed` flag on the
  smoke binaries + proper held-out eval set (HumanEval / MBPP). The
  Stage H smoke shows the protocol works; reproducing Phase 17-20
  numbers via this stack is its own measurement run.
- **`AgenticGeneratorActor<M>`**: Phase 4 tool-use loop still
  hardcodes `ActorRef<ModelActor>`. Generic-ifying it is mechanical
  (mirror Stage E) but outside the Phase 17-20 recipe path.
- **DPO via Qwen trainer**: `QwenTrainerActorHandle::train(Dpo)` bails.
  Phase 11/14 settled DPO as non-winning, so this isn't a priority.

## Phase 21 stage roadmap — complete

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 inference | ✅ (`acfdc5d`) |
| F | Candle-native Qwen2 LoRA training standalone | ✅ (`8367d2a`) |
| B | Substrate scale-up + pass@k lift | ✅ (`8b925a1`) |
| E | Generic Evaluator/Generator | ✅ (`3ead388`) |
| E.next | QwenTrainerActor | ✅ (`ec00809`) |
| E.next.next | Multi-actor multi-round demo + merge | ✅ (`8105007`) |
| **H** | **`TrainerHandle` trait + supervisor wiring** | ✅ (this commit) |
| G | RL with pass@k reward | deferred |

## Files

- `llm-actors/src/trainer_handle.rs` — trait + two handles (new)
- `llm-actors/src/supervisor.rs` — `RoundActors.trainer: Arc<dyn TrainerHandle>`,
  trait dispatch in `run_round`
- `llm-actors/src/lib.rs` — re-exports
- `llm-actors/examples/phase21_h_smoke.rs` — full pipeline against Qwen
- `llm-actors/examples/{self_improve_round,self_improve_korean,self_improve_rust,phase21_multi_round_smoke}.rs`
  — wrap `trainer_ref` in `TrainerActorHandle`
- `llm-actors/Cargo.toml` — new example registration
- `docs/phase21-stage-h.md` (this)

## See also

- `docs/phase21-stage-e-next-next.md` — hand-rolled multi-actor loop
- `docs/phase21-stage-e-next.md` — QwenTrainerActor
- `docs/phase21-stage-e.md` — generic Evaluator/Generator/RoundActors
- `docs/phase21-stage-d.md` — QwenModelActor inference
- `docs/phase21-stage-f.md` — Qwen2 LoRA training (standalone)
- `docs/phase20-closeout.md` — Phase 17-20 recipe this Pekko bridge now serves
