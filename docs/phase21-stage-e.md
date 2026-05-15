# Phase 21 Stage E — generic-ify EvaluatorActor / GeneratorActor / RoundActors

Stage D landed `QwenModelActor` so the same Qwen2.5-Coder-0.5B model
Phase 14-20 used could be served from the Rust actor stack. But the
type system stopped right there: `EvaluatorActor.model:
ActorRef<ModelActor>` was hardcoded, so `ActorRef<QwenModelActor>`
couldn't plug in even though both actors share the `ModelMessage`
enum. Stage A's pass@k mechanism worked only against `ModelActor`
in the supervisor pipeline.

Stage E closes that gap. After this commit `EvaluatorActor`,
`GeneratorActor`, `RoundActors`, `run_round`, and `run_multi_round`
are all generic over `M: Actor<Message = ModelMessage>` with a
default of `M = ModelActor` — every existing call site keeps
working untouched, and new code can spell `EvaluatorActor::<QwenModelActor>`
to drive the production model.

## What's in this commit

### Library changes (generic-ification)

`llm-actors/src/evaluator_actor.rs`:
- `EvaluatorActor<M = ModelActor>` where `M: Actor<Message = ModelMessage>`
- Inherent + Actor impls re-bounded over M

`llm-actors/src/generator_actor.rs`:
- `GeneratorActor<M = ModelActor>` where `M: Actor<Message = ModelMessage>`
- Inherent + Actor impls re-bounded over M

`llm-actors/src/supervisor.rs`:
- `RoundActors<M = ModelActor>` — `model: ActorRef<M>`,
  `generator: ActorRef<GeneratorActor<M>>`, `evaluator: ActorRef<EvaluatorActor<M>>`.
  `verifier` / `curator` / `trainer` don't hold model refs and stay
  non-generic.
- `pub async fn run_round<M>(actors: &RoundActors<M>, ...)`
- `async fn ask_eval<M>(evaluator: &ActorRef<EvaluatorActor<M>>, ...)`
- `pub async fn run_multi_round<M, F>(actors: &RoundActors<M>, ...)`

### Stage E smoke (`phase21_e_smoke`)

Demonstrates the genericization end-to-end:
1. Load `Qwen2.5-Coder-0.5B` as `QwenModelActor`
2. Wrap the HF tokenizer in `nanogpt_rs::Tokenizer::Bpe(...)` (the
   existing tokenizer enum already supports BPE — Stage E just exercises
   that branch)
3. Build `EvaluatorActor::<QwenModelActor>::new(...)` — type works
4. Spawn it in an `ActorSystem`
5. Send `EvaluatorMessage::Eval` with `passk=1` and `passk=5`

Output:
```
[Phase21E] EvaluatorActor<QwenModelActor> spawned
[Phase21E] passk= 1  pass-rate=1.000  (12/12)  eval_sampling(temp=0, topk=1)
    sample 0  prompt="def fibonacci(n):"
    completion="\n    if n == 0:\n        return 0\n    elif n == 1:..."
[Phase21E] passk= 5  pass-rate=1.000  (12/12)  eval_sampling(temp=0.8, topk=40)
    sample 0  completion="\n    if n <= 1:\n        return n\n    else:\n        return fibonacci..."
    sample 1  completion="\r\n    a = 1\r\n    b = 1\r\n    if n == 1:\r\n        print(a)"

phase21_e_smoke: PASS
```

Trivial verify (completion contains `"return"`) so the smoke focuses
on wiring, not realism. The sampling-diverse completions at `passk=5`
(recursive vs iterative) prove the per-(prompt, k) seed override from
Stage A flows through QwenModelActor correctly.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **144 tests** (no change —
  all existing tests still pass with default `M = ModelActor`)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_e_smoke`: `EvaluatorActor<QwenModelActor>` produces
  diverse passk=5 samples, pass-rate 1.000 on trivial verify

## What stays non-generic (and why)

- `AgenticGeneratorActor` — holds `ActorRef<ModelActor>` too, but
  Phase 17-20 path doesn't use it (it's the Phase 4 tool-use loop).
  Generic-ifying is straightforward but punted to keep this commit
  focused.
- `InferenceServerActor` — same situation; it's the HTTP frontend.
- `EnsembleActors` — `Vec<ActorRef<ModelActor>>`; ensemble experiments
  (Phase 5) didn't survive Phase 13 retraction.
- `TrainerActor` — doesn't hold a model ActorRef at all; trains via
  `spawn_blocking` directly. Qwen LoRA training (Stage F) ships its
  own train function; integrating it into TrainerActor is the
  natural Stage E.next.

## What this unlocks

Combine Stage E + Stage D + Stage A + Stage C:

```rust
let qwen = QwenModelActor::from_snapshot_dir(...)?;
let model_ref = system.spawn(qwen, "model").await?;
let evaluator = EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), ...);
let evaluator_ref = system.spawn(evaluator, "evaluator").await?;
let generator = GeneratorActor::<QwenModelActor>::new(model_ref.clone(), ...);
let generator_ref = system.spawn(generator, "generator").await?;

let actors = RoundActors::<QwenModelActor> {
    model: model_ref,
    generator: generator_ref,
    evaluator: evaluator_ref,
    verifier: verifier_ref,
    curator: curator_ref,
    trainer: trainer_ref,
};
let reports = run_multi_round(&actors, MultiRoundConfig::new(3, base), |r, rep| {
    println!("round {r}: ...");
}).await?;
```

This is the **Phase 17-20 multi-round SFT recipe wired through the
Pekko actor stack against the real Qwen production model**. The
remaining gap is QwenTrainerActor (Stage E.next) — TrainerActor
currently trains nanogpt_rs::GPT via train_from. Pairing with
Stage F's `train_qwen_lora_step` is short follow-on work.

## Phase 21 stage roadmap (post Stage E)

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 + QwenModelActor (inference) | ✅ (`acfdc5d`) |
| F | Candle-native Qwen2 LoRA training | ✅ (`8367d2a`) |
| B | Substrate scale-up + pass@k lift | ✅ (`8b925a1`) |
| **E** | **Generic Evaluator/Generator over `Actor<Message=ModelMessage>`** | ✅ (this commit) |
| E.next | QwenTrainerActor (Trainer side of the Pekko bridge) | next-up |
| G | RL with pass@k reward | deferred |

After E, the only thing keeping Phase 17-20 multi-round SFT from
running end-to-end against real Qwen in the Pekko stack is the
trainer-side integration. Stage E.next is a small wrapper.

## Files

- `llm-actors/src/evaluator_actor.rs` — `<M>` generic
- `llm-actors/src/generator_actor.rs` — `<M>` generic
- `llm-actors/src/supervisor.rs` — `RoundActors<M>` + helpers generic
- `llm-actors/examples/phase21_e_smoke.rs` — new E2E demonstration
- `llm-actors/Cargo.toml` — example registration
- `docs/phase21-stage-e.md` (this)

## See also

- `docs/phase21-stage-a.md` — pass@k mechanism
- `docs/phase21-stage-c.md` — `run_multi_round` helper
- `docs/phase21-stage-d.md` — `QwenModelActor` inference
- `docs/phase21-stage-f.md` — Qwen2 LoRA training
- `docs/phase21-stage-b.md` — pass@k lift at Rust-native scale-up
