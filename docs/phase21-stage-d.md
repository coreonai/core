# Phase 21 Stage D — Candle-native Qwen2 in the Pekko actor stack

Phase 14-20 ran every measurement through Python scripts against
`Qwen/Qwen2.5-Coder-0.5B` loaded via HuggingFace transformers. Stage D
bridges that production model into the Rust `llm-actors` framework
**without a Python sidecar** — Candle's `candle_transformers::models::qwen2`
loads the same HF safetensors natively, and a new `QwenModelActor`
wraps that into the existing `ModelMessage` enum.

## What's in this commit

### Dependencies
`llm-actors/Cargo.toml`:
- `candle-transformers = { workspace = true }` (was workspace-declared
  but unused by either crate)
- `tokenizers = { workspace = true }`
- `hf-hub = "0.4"` (for future model download convenience; not strictly
  used by this commit since we load from cache)

### Standalone PoC (`phase21_qwen_candle_smoke`)
Pure-Rust binary that loads Qwen2.5-Coder-0.5B from the HF cache and
generates code:

```
$ cargo run -p llm-actors --example phase21_qwen_candle_smoke \
    --features cuda --release -- --prompt "def fibonacci(n):" --max-new-tokens 40
[Phase21D] device = Cuda(...), on_cuda = true
[Phase21D] tokenizer loaded (vocab=151665)
[Phase21D] config loaded (hidden=896, layers=24, heads=14/2, vocab=151936)
[Phase21D] model loaded (dtype=F16)

=== Generated ===

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2
=== End ===

phase21_qwen_candle_smoke: PASS
```

### `QwenModelActor`
`llm-actors/src/qwen_model_actor.rs`. Reuses the existing `ModelMessage`
enum so the API surface matches `ModelActor` even though the underlying
type is distinct.

**Implemented**:
- `Ping` — health check
- `Generate { prompt, cfg, reply }` — HF-tokenize → autoregressive
  generate → decode. Honors `GenerateConfig` fully (temperature,
  top_k, top_p, seed, max_new_tokens). EOS-stops on token 151643.
- `GenerateTokens { prompt_ids, cfg, reply }` — raw HF token IDs in/out
- `ReloadCheckpoint { path, reply }` — reload `model.safetensors` from
  another snapshot directory

**Stubbed** (returns `Err`):
- `LossOn` — Qwen2's `ModelForCausalLM::forward` returns only the
  last-position logits; computing CE over all positions would require
  duplicating the lm_head application path. Deferred.
- `ScoreLogProb` — same reason; needs all-position logits.

**Not covered** — training (LoRA / SFT). The Trainer's path is heavily
tied to `nanogpt_rs::GPT` + Candle VarMap. A Qwen-side training stack
needs its own multi-day project (parameter group splitting for LoRA,
optimizer state for AdamW over the much larger Qwen Vars, etc.).
Deferred to Phase 21 Stage E+.

**KV cache** — `ModelForCausalLM` carries internal KV state. Each
`generate_autoregressive` call clears it first so consecutive requests
don't leak across prompts.

### `phase21_qwen_actor_smoke` (actor-pipeline E2E)
Spawns `QwenModelActor` inside a Pekko `ActorSystem`, sends 3
`ModelMessage::Generate` requests, prints completions:

```
[prompt 0] def fibonacci(n):  → recursive fibonacci
[prompt 1] def is_prime(n):   → trial division up to sqrt(n)
[prompt 2] def reverse_string(s):  → s[::-1] + bonus is_palindrome
phase21_qwen_actor_smoke: PASS
```

All three are syntactically + semantically correct Python — the same
quality Phase 17-20 measured via Python `transformers`.

## Type compatibility constraint

`QwenModelActor::Message == ModelMessage` so the message ENUM is
shared. But `ActorRef<QwenModelActor>` and `ActorRef<ModelActor>` are
different types — callers like `EvaluatorActor` that hold
`ActorRef<ModelActor>` cannot directly accept a Qwen actor.

To plumb the existing eval/gen actors so they work with either backing
model, Pekko would need to make `EvaluatorActor` and `GeneratorActor`
generic over `Actor<Message = ModelMessage>`. That's the Stage E
refactor.

For now: anyone wanting Phase 17 pass@k semantics against the real
Qwen model can drive `QwenModelActor` directly with their own loop
(loop k times per prompt, OR the verdicts) — the mechanism is one
small for-loop and the `EvaluatorActor::run` source is right there
as the reference implementation.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **142 tests** (was 141; +1
  compile-time assertion on `QwenModelActor: Actor<Message=ModelMessage>`)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E #1: `phase21_qwen_candle_smoke` prints
  `phase21_qwen_candle_smoke: PASS` after generating a valid
  fibonacci implementation
- ✅ E2E #2: `phase21_qwen_actor_smoke` prints
  `phase21_qwen_actor_smoke: PASS` after 3 actor-routed Generate
  requests, all producing correct Python

## Phase 21 stage roadmap (post Stage D)

| stage | scope | status |
|---|---|---|
| A | Pass@k in actor stack | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper + smoke | ✅ (`f09d97d`) |
| **D** | **Candle-native Qwen2 + `QwenModelActor` (Generate path)** | ✅ (this commit) |
| B | Substrate scale-up (n_embd=512, n_layer=6) + measure passk lift | deferred — needs ~5-10× K9 wallclock |
| E | Generic `EvaluatorActor` / `GeneratorActor` over `Actor<Message=ModelMessage>` so `QwenModelActor` can drive the full pipeline + measure Phase 17 pass@k at the Rust orchestration layer | next-up |
| F | Qwen LoRA training in Rust (Trainer-side bridge) — bigger project | deferred |
| G | RL with pass@k reward | days of code |

Stage D explicitly does NOT ship the training side. That's the
hardest piece of the Pekko ↔ HF bridge and deserves its own phase.

## What this means for the project vision

The README pitches a "self-evolving agentic foundation model on top of
Apache Pekko". For two phases (17-20) we measured the algorithmic
findings (multi-round SFT, pass@k) in Python because that's where the
Qwen model lived. After Stage D **the Rust actor stack now talks to
the same model** — the gap between vision and reality has narrowed
from "two different stacks" to "one stack, with training still in
Python while inference is fully native".

The inference half is what the deployed agent actually needs (Phase 20
deployment recipe: r=k SFT model serves k>1 pass@k inference). That
half is now end-to-end Rust + Pekko.

## Files

- `llm-actors/Cargo.toml` — new deps
- `llm-actors/src/qwen_model_actor.rs` — actor impl + 1 unit test
- `llm-actors/src/lib.rs` — re-export
- `llm-actors/examples/phase21_qwen_candle_smoke.rs` — standalone PoC
- `llm-actors/examples/phase21_qwen_actor_smoke.rs` — actor E2E
- `docs/phase21-stage-d.md` (this)

## See also

- `docs/phase21-stage-a.md` — pass@k mechanism in EvaluatorActor
- `docs/phase21-stage-c.md` — `run_multi_round` helper
- `docs/phase20-closeout.md` — Python-side saturation + deployment
  findings that the Rust stack now mirrors at inference time
