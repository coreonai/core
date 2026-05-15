# Phase 21 Stage F — Qwen2 LoRA training in Rust (Candle-native)

Stage D landed inference: `QwenModelActor` serves the same Qwen2.5-Coder-0.5B
model Phase 14-20 used, with no Python sidecar. Stage F lands **training**:
a Candle-native Qwen2 LoRA implementation that compiles, runs, and
demonstrably reduces loss via gradient descent.

## What's in this commit

### `llm-actors/src/qwen2_lora.rs`
Fork of `candle_transformers::models::qwen2` with:

1. **LoRA injection on `q_proj` + `v_proj`** — matches Phase 14-20's
   PEFT recipe (`target_modules = ["q_proj", "v_proj"]`, r=16, α=32).
   Frozen base linear weights load from the upstream safetensors; LoRA
   `(a, b)` adapter Vars live in a separate VarMap so the optimizer
   only touches them.
2. **`ModelForCausalLM::forward_train`** returns all-position logits
   `(B, T, V)` instead of the last position only (which the upstream
   `forward` narrows to). Required for next-token-prediction
   cross-entropy loss over the full sequence.
3. **Training-time `_slow` ops dispatch** — `rotary_emb::rope_slow`,
   `ops::rms_norm_slow`, and `ops::softmax(..., D::Minus1)` replace
   the no-backward-pass kernels (`rope`, `rms_norm`, `softmax_last_dim`)
   along the training path. Without this the gradient chain breaks
   silently — see "Debugging notes" below.
4. **`train_qwen_lora_step`** — forward all positions → cross-entropy on
   shifted next-token targets → backward → AdamW step over the LoRA
   VarMap. Returns the pre-step loss for trajectory logging.
5. **`lora_grad_norms` diagnostic** — backward only, no step; reports
   gradient L2 norms over the supplied Vars. Used by the smoke binary
   to gate training start on the gradient flow actually working.

### `llm-actors/examples/phase21_qwen_lora_smoke.rs`
End-to-end smoke:
1. Load Qwen2.5-Coder-0.5B from HF cache (same snapshot Stage D uses)
2. Build the LoRA model (r=16, α=32) over a fresh VarMap
3. Diagnostic: run one forward+backward, count how many LoRA Vars
   receive gradients. Abort if none do (== gradient chain broken)
4. Run N steps of `train_qwen_lora_step` over a fixed `(input, target)`
5. Assert final loss < initial loss

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **144 tests** (was 142;
  +1 `lora_config_default_matches_phase14_20_recipe`,
  +1 `isolated_lora_adapter_gets_gradient_through_backprop`)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E smoke @ 8 steps:
  ```
  step 0  loss = 0.8226
  step 1  loss = 0.7146
  step 2  loss = 0.6346
  step 3  loss = 0.5633
  step 4  loss = 0.4974
  step 5  loss = 0.4316
  step 6  loss = 0.3832
  step 7  loss = 0.3530

  loss: 0.8226 → 0.3530  Δ = -0.4696
  phase21_qwen_lora_smoke: PASS
  ```
  Loss reduced by **57%** over 8 AdamW steps with the Phase 14-20
  hyperparameters (lr=2e-4, β=0.9/0.999, weight_decay=0).

## LoRA gradient analysis

Across 96 LoRA Vars (24 layers × 4 = q_a/q_b/v_a/v_b each):

| family | shape | count | grad on step 0 |
|---|---|---:|---|
| q_a, v_a | `(rank=16, in=896)` | 48 | exactly 0 — `lora_b` init is zero, so `∂L/∂a = b·∂L/∂y` is zero |
| q_b | `(out=896, rank=16)` | 24 | non-zero — `∂L/∂b = a·x · ∂L/∂y` flows |
| v_b | `(out=128, rank=16)` | 24 | non-zero |

This is the **expected LoRA cold-start behavior**: `b` updates on step 0
even though `delta = b·a·x = 0`; after `b` ≠ 0 by step 1, `a` starts
updating too. Loss decreases monotonically from there.

## Debugging notes — why this took longer than expected

Initial run had loss stuck at 0.8226 across all steps. Diagnostic showed
**zero of 96 LoRA Vars received gradients**. Root cause: three of
candle-nn's "fast" kernels are forward-only:

| kernel | underlying call | backward defined? |
|---|---|---:|
| `rotary_emb::rope` | `apply_op3_no_bwd(RotaryEmb)` | ❌ |
| `ops::softmax_last_dim` | `apply_op1_no_bwd(SoftmaxLastDim)` | ❌ |
| `ops::rms_norm` (used by `RmsNorm::Module`) | `apply_op2_no_bwd(RmsNorm)` | ❌ |

Inference (the original `qwen2` module) uses all three and is fine —
it never runs backward. Training does, and silently loses the gradient
chain at every layer. The fix is the `_slow` variants on the training
path (`rope_slow`, `softmax`, `rms_norm_slow`) which compose pure
tensor ops with proper backward support.

This is a Candle 0.10 gotcha that's not well-advertised. The forks
should consider deferring to the slow ops whenever training is the
goal, or candle could mark these clearly in the rustdoc. Documenting
here so future Phase 22+ work on Candle-native training avoids re-discovering it.

## What this does NOT do

- **Adapter save/load** to a separate `.safetensors` file — the
  in-memory VarMap is enough for the smoke. A persistence helper using
  `VarMap::save` and `VarMap::load` would be one short function.
- **Actor integration**. A `QwenTrainerActor` would mirror Stage D's
  `QwenModelActor` for training requests. Deferred to a follow-up
  stage — pairs naturally with the existing `TrainerActor` flow.
- **F16 training**. F32 throughout because rank-16 LoRA gradients at
  lr=2e-4 are numerically too coarse in F16. Mixed-precision (F16
  forward, F32 LoRA accumulators) is doable but deferred.
- **The full Phase 17-20 recipe**. The smoke is wiring-grade — one
  small synthetic batch — not a multi-round SFT replication. Phase 22+
  can plug `train_qwen_lora_step` into `supervisor::run_multi_round`
  (Stage C) for a full Rust-side multi-round SFT against real
  HumanEval/MBPP corpora.

## Phase 21 stage roadmap (post Stage F)

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper + smoke | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 + `QwenModelActor` (inference) | ✅ (`acfdc5d`) |
| **F** | **Qwen2 LoRA training in Rust (Candle-native)** | ✅ (this commit) |
| B | Substrate scale-up + measure passk lift | deferred |
| E | Generic Evaluator/Generator over `Actor<Message=ModelMessage>` | next-up |
| G | RL with pass@k reward | deferred |

After F, the **inference + training** halves of the Pekko vision are
both Rust-native. Stage E (generic-ify the eval/gen actors) closes the
last gap so existing `supervisor::run_round` orchestration drives
real Qwen instead of nanogpt_rs::GPT.

## Files

- `llm-actors/src/qwen2_lora.rs` — forked qwen2 with LoRA + slow-op training path
- `llm-actors/src/lib.rs` — re-export
- `llm-actors/examples/phase21_qwen_lora_smoke.rs` — E2E smoke
- `llm-actors/Cargo.toml` — example registration
- `docs/phase21-stage-f.md` (this)

## See also

- `docs/phase21-stage-d.md` — inference side (`QwenModelActor`)
- `docs/phase21-stage-c.md` — multi-round orchestration helper
- `docs/phase21-stage-a.md` — pass@k mechanism
- `docs/phase20-closeout.md` — the Python-side findings this Rust
  training stack is now positioned to reproduce
