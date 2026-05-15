# Phase 21 Stage E.next.next — full Pekko-bridge multi-round demo

Stage E.next landed `QwenTrainerActor` for training. Stage D landed
`QwenModelActor` for inference. Stage E landed the generic
`EvaluatorActor<M>`. But **inference didn't see training** — the two
actors hold separate model copies, and the upstream `qwen2`
module (`QwenModelActor`'s backing) doesn't know how to load LoRA
adapters at runtime.

Stage E.next.next closes that loop by **merging LoRA into base
safetensors** post-training. The merged file is drop-in compatible
with the upstream loader, so `QwenModelActor::ReloadCheckpoint` picks
it up without any LoRA-awareness on the inference side. A new
end-to-end smoke wires all three actors and demonstrates the
Phase 17-20 recipe shape (train → save merged → reload → eval) on
the real Qwen2.5-Coder-0.5B production model.

## What's in this commit

### Library

`llm-actors/src/qwen2_lora.rs` — new `save_merged_lora` helper:

```rust
pub fn save_merged_lora(
    base_safetensors_path: &Path,
    lora_map: &VarMap,
    cfg: &Config,
    lora_cfg: LoraConfig,
    device: &Device,
    out_path: &Path,
) -> Result<()>
```

Reads every tensor from the original frozen-base safetensors, then for
each layer's `q_proj` and `v_proj` adds the LoRA delta
`B @ A * (α / r)` into the `weight` tensor (cast to the base's dtype).
All other tensors (k_proj / o_proj / norms / embed_tokens / lm_head)
pass through unchanged. The output is structurally identical to the
input — same keys, same dtypes, same shapes — so the upstream
`candle_transformers::models::qwen2` loader treats it as just another
Qwen2 checkpoint.

`llm-actors/src/qwen_trainer_actor.rs` — new message variant:

```rust
QwenTrainerMessage::SaveMergedCheckpoint {
    base_path: PathBuf,
    out_path: PathBuf,
    reply: oneshot::Sender<Result<()>>,
}
```

`QwenTrainerActor` now also remembers `lora_cfg` (rank, α) so the merge
can reconstruct the `α / r` scale. Plumbed through the constructor;
no callsite changes.

### Smoke (`phase21_e_next_next_smoke`)

Spawns three actors in one `ActorSystem`:
- `QwenModelActor` (F16 on CUDA, upstream qwen2 loader)
- `QwenTrainerActor` (F32, qwen2_lora with LoRA r=16, α=32, lr=2e-4)
- `EvaluatorActor::<QwenModelActor>` (Stage E genericization)

Loop:
1. Baseline eval at passk=1 and passk=5 (pre-train)
2. For each round:
   - `Train { texts, train_steps }` via QwenTrainerActor
   - `SaveMergedCheckpoint { base_path, out_path }` — emits a
     base-compatible safetensors with LoRA baked in
   - `ModelMessage::ReloadCheckpoint { path: out_path }` on
     QwenModelActor — inference now reflects training
   - Eval-after at passk=1 and passk=5

Output:
```
[Phase21E.next.next] QwenTrainerActor built  lora_params = 1081344
[Phase21E.next.next] 3 actors spawned: model + trainer + evaluator
[Phase21E.next.next] baseline  pass@1=1.000  pass@5=1.000

=== Round 0 ===
  train loss: 0.522 → 0.792 (per-step over 6 steps; cycles 3 texts)
  merged checkpoint saved  size=988097792 bytes
  QwenModelActor reloaded from merged checkpoint
  eval-after  pass@1=1.000  pass@5=1.000

=== Round 1 ===
  train loss: 0.342 → 0.458 (per-step over 6 steps)
  merged checkpoint saved  size=988097792 bytes
  QwenModelActor reloaded from merged checkpoint
  eval-after  pass@1=1.000  pass@5=1.000

phase21_e_next_next_smoke: PASS
```

Pass-rate stayed at 1.000 across the whole run because the smoke uses
a trivial domain (completion contains `"return"`) — Qwen always
produces a function body with `return`, so the eval is saturated.
**The Stage E.next.next deliverable is the multi-actor protocol
running end-to-end**, not a benchmark signal. The 988 MB merged
checkpoint is identical-shape to the 988 MB base (just with q/v
weights perturbed by the LoRA delta).

Per-step train loss values are NOT monotonically decreasing because
the trainer cycles through 3 different texts; each step's loss is
measured against a different prompt. Stage F's standalone smoke
(single text, 8 steps) is the clean monotonic curve; this smoke shows
the multi-text round-robin.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **145 tests** (no change)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_e_next_next_smoke` PASS — 2 rounds of
  Train + SaveMergedCheckpoint + ReloadCheckpoint + Eval through
  3 cooperating actors

## Project-vision status

After Stage E.next.next, the **Phase 17-20 self-improvement recipe
shape** (Gen-Train-Eval round with multi-round chaining) runs
end-to-end through the Pekko actor stack against the real
Qwen2.5-Coder-0.5B production model. Concretely:

| step | actor |
|---|---|
| Eval-before (pass@1 / pass@k) | `EvaluatorActor::<QwenModelActor>` |
| Train on a corpus              | `QwenTrainerActor` |
| Save trained model             | `QwenTrainerActor::SaveMergedCheckpoint` |
| Reload trained model           | `ModelMessage::ReloadCheckpoint` to `QwenModelActor` |
| Eval-after                     | same `EvaluatorActor` |

What's still NOT integrated (deferred):
- **`supervisor::run_round` / `run_multi_round`** still use
  `TrainerActor` (nanogpt_rs) via `RoundActors.trainer`. Wiring
  `QwenTrainerActor` in requires `RoundActors<M, T>` generic over the
  trainer too, OR a trainer-step trait both impls satisfy. Both are
  bigger surgery; the loop in this smoke is hand-rolled instead.
- **GeneratorActor** isn't part of this smoke. The Phase 17-20
  recipe's "Gen" phase needs to drive `QwenTrainerActor.Train`'s
  corpus — currently the smoke uses a hardcoded text list. A full
  integration would have `GeneratorActor<QwenModelActor>` emit
  trajectories → `VerifierActor` → `CuratorActor` → corpus →
  trainer. All those actors are Stage E generic-ready; the only
  remaining piece is the trainer integration (see above).

## Phase 21 stage roadmap (post Stage E.next.next)

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 inference | ✅ (`acfdc5d`) |
| F | Candle-native Qwen2 LoRA training standalone | ✅ (`8367d2a`) |
| B | Substrate scale-up + pass@k lift | ✅ (`8b925a1`) |
| E | Generic Evaluator/Generator | ✅ (`3ead388`) |
| E.next | QwenTrainerActor (training side actor) | ✅ (`ec00809`) |
| **E.next.next** | **Multi-actor multi-round demo + merge** | ✅ (this commit) |
| H | `RoundActors<M, T>` + `run_multi_round` wiring QwenTrainerActor | next-up |
| G | RL with pass@k reward | deferred |

Stage H closes the supervisor integration — at that point the full
Phase 17-20 recipe (with proper Generator → Verifier → Curator pipe)
runs through `run_multi_round`, not just a hand-rolled smoke loop.

## Files

- `llm-actors/src/qwen2_lora.rs` — `save_merged_lora` helper
- `llm-actors/src/qwen_trainer_actor.rs` — `SaveMergedCheckpoint`
  message + `lora_cfg` on actor
- `llm-actors/examples/phase21_e_next_next_smoke.rs` — new E2E demo
- `llm-actors/Cargo.toml` — example registration
- `docs/phase21-stage-e-next-next.md` (this)

## See also

- `docs/phase21-stage-e.md` — generic Evaluator/Generator/RoundActors
- `docs/phase21-stage-e-next.md` — QwenTrainerActor
- `docs/phase21-stage-d.md` — QwenModelActor inference
- `docs/phase21-stage-f.md` — standalone train_qwen_lora_step
