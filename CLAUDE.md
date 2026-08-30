# CLAUDE.md — workLLM Repo Context

This file orients future Claude Code sessions to this repository. The
README.md is the user-facing description; this is the engineer-facing
working knowledge that doesn't fit there.

## What this repo is

A Rust-only implementation of a self-evolving agentic foundation model,
built across 11 phases on top of the user's `pekko-rust` (Apache Pekko port,
at `../AgenticAI/rust_pekko/`). The phases compose: each phase ships
infrastructure that the next phase exercises.

The vision is **infrastructure**, not a state-of-the-art model. Where this
matters: do not "improve" the toy task ceilings (arithmetic ~50%, Korean
training loss ~7.0). They are dataset-/scale-limited, not bug-limited.
Treat the existing 60+ unit tests as the contract.

## Repo layout

```
workLLM/
├── Cargo.toml                  # workspace; pulls pekko-actor by path dep
├── nanogpt-rs/                 # GPT model + tokenizer + training + EWC + sampling
│   └── src/
│       ├── config.rs           # GPTConfig (12 axes), nano_*  presets
│       ├── model.rs            # GPT, Block, MLP, CausalSelfAttention, FeedForward (MoE), Norm wrapper, LoRA
│       ├── tokenizer.rs        # CharTokenizer + HF BPE training + load
│       ├── data.rs             # TokenDataset
│       ├── train.rs            # AdamW + cosine LR + EWC + LoRA freeze + distillation
│       ├── ewc.rs              # WeightAnchor (uniform + diagonal Fisher)
│       └── generate.rs         # temperature/top-k/top-p (greedy = temp 0)
├── llm-actors/                 # pekko-rust actor wrappers
│   └── src/
│       ├── model_actor.rs      # owns VarMap + GPT, hot-reload via ReloadCheckpoint
│       ├── trainer_actor.rs    # spawn_blocking continual fine-tune (LoRA-aware)
│       ├── curator_actor.rs    # priority replay buffer
│       ├── generator_actor.rs
│       ├── verifier_actor.rs
│       ├── evaluator_actor.rs
│       ├── supervisor.rs       # 1-round Gen→Verify→Curate→Train→Reload→Eval
│       ├── evolution.rs        # SearchSpace + Variant + EvolutionRunner
│       ├── tools/              # Tool trait + ToolRegistry + ArithmeticTool
│       ├── tool_executor_actor.rs
│       ├── agentic_generator_actor.rs   # multi-turn tool dispatch
│       ├── inference_server_actor.rs    # transport-neutral RPC actor
│       ├── inference_http.rs            # axum HTTP frontend
│       └── domain/
│           ├── arithmetic.rs   # SeedMode (Full/NoCarry/None)
│           ├── tool_use.rs     # ToolUseArithmeticDomain
│           └── rust_code.rs    # cargo-build verifier
├── data/                       # gitignored: corpora, tokenizers
├── checkpoints/                # gitignored: safetensors
└── README.md
```

## Build & run

### Required environment

```bash
# CUDA driver is 555 → CUDA 12.5 toolkit. The default cuda-12.9 toolkit
# generates PTX the driver rejects (DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)).
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH
```

### Build patterns

```bash
# CPU-only (works on every platform):
cargo build --workspace --release

# CUDA (one A100, default):
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
    cargo build -p llm-actors --example evolve_arithmetic --features cuda --release

# Pick a specific GPU when others are busy:
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/...
```

### Tests

```bash
cargo test --workspace            # ~60 unit tests, all CPU
```

## Phase-by-phase mental model

Each phase composes the previous. **Read order if you only have time for some:**
phase 1 (model basics) → phase 3 (NAS) → phase 4 (tools + agent loop). Phase 2
infrastructure is exercised by 4.

### Phase 1: model

`nanogpt-rs::model::GPT` is the heart. Architecture is configurable across
**12 axes** (`GPTConfig`):

```rust
n_layer, n_head, n_embd, block_size, ffn_mult,
use_rope, kv_group (n_kv_head), n_experts,
activation (Gelu/SwiGlu/GeGlu), weight_tying,
norm_kind (LayerNorm/RmsNorm), norm_position (Pre/Post)
```

Default `nano_50m()` is the Phase-3-discovered Llama recipe (RoPE + 4× GQA
+ SwiGLU + RmsNorm-Pre + untied head, ~46.8M params at vocab 32k).

**Critical bug fixed in phase 4 work:** `generate.rs::sample_logits` used
to do `logits / cfg.temperature` even when `temperature == 0.0`, which
gave ±∞ logits and silently collapsed sampling to the first non-negative
logit. Greedy decode now branches early and returns argmax.

### Phase 2 & 2.5: self-improvement loop

Six actors (`Generator`, `Verifier`, `Curator`, `Trainer`, `Evaluator`,
`Supervisor`) compose into one round:
`Gen → Verify → Curate → Train → Reload → Eval`. Curator supports priority
replay (`SampleMode::Priority { recency_decay }`).

`Domain` trait abstracts the task. Three impls: `ArithmeticDomain` (toy),
`ToolUseArithmeticDomain` (Phase 4), `RustCodeDomain` (cargo build/run as
verifier).

### Phase 3: NAS / evolution (×7 turns)

`evolution.rs::EvolutionRunner` runs `population_size × generations` evals,
each variant trained from scratch and scored by domain pass-rate. Multi-GPU
via `tokio::task::spawn_blocking` + `JoinSet` round-robined across
`Cuda(i % n_gpus)`.

The 12-axis search space was added one axis at a time across 7 turns. By
the final turn, evolution **independently rediscovered the Llama-2 recipe**
(RoPE + GQA + SwiGLU + RmsNorm-Pre + untied head) — there's no human-coded
preference for it.

Each `Variant` carries an `origin: VariantOrigin` (`Random`, `Mutated`,
`Crossover`, `Elite`) so generation lineage is fully traceable.

### Phase 4: tool use, agent loop, distillation, EWC, LoRA (×11 turns)

- `tools::Tool` trait; registry dispatch.
- `ToolExecutorActor` runs handlers; `AgenticGeneratorActor` does the
  multi-turn `generate → parse_first_tool_call → dispatch → splice_result
  → continue` loop. **Critical invariant:** `parse_first_tool_call` skips
  tool calls whose body contains `=`, because that's the marker
  `splice_result` writes — without this, the loop would re-fire forever
  on the just-resolved call.
- Catastrophic-forgetting comparison **fully ablated**: plain fine-tune,
  replay mixing (ER), uniform-Fisher EWC, real diagonal-Fisher EWC, LoRA
  (c_attn-only and per-Linear with `freeze_base=true`). The full matrix
  is in README.

### Phase 1 epilogue: KoWiki

Real Korean Wikipedia training validated. `extract_kowiki` streams a
`bzcat` pipe through `quick-xml` + regex cleanup; `train_kowiki` trains a
50M Llama-recipe model. Loss plateau at ~7.0 against `ln(16K)=9.68` —
expected at this scale. For comparison, `compare_tokenizers` benchmarks
our 16K BPE against the 30K Polyglot-Ko HF tokenizer (ours wins on
in-domain compression, HF wins on coverage).

## Common gotchas / "don't do this" notes

1. **Never** drop `mut` on `VarMap` when you need `varmap.load(path)`. The
   error is misleading ("cannot borrow as mutable") because most other
   VarMap methods take `&self`.

2. **Never** load a checkpoint that uses a different architecture without
   matching `GPTConfig` exactly. Param names depend on `weight_tying`,
   `lora_rank`, `n_kv_head`, etc. Use the `.cfg.json` saved alongside.

3. **Never** train Korean with the original `nano_50m` if you're
   referencing the latest version of `config.rs`. As of phase-1-epilogue,
   `nano_50m` was retrofitted with Phase 3's Llama recipe (RoPE + GQA-2 +
   SwiGLU + RmsNorm-Pre + untied) and `vocab_size: 32000`. Older saved
   checkpoints are NOT loadable into the new preset — pin the config.

4. **Never** assume `train_loss` reported by `TrainConfig::smoke` is a
   smoothed metric. It's the LAST batch's loss only. Use
   `eval_interval > 0` + a `val_ds` to get reliable measurements.

5. **Always** quote your CUDA env: builds without `CUDA_HOME=/usr/local/cuda-12.5`
   pick up cuda-12.9 PTX that the 555 driver refuses at runtime. Compile
   succeeds; first kernel launch panics. Re-run with the env variable.

6. **Always** `cargo build` from the workspace root, not from
   `data/kowiki/` or other subdirs. Several scripts `cd` into data dirs
   to call `curl`; subsequent cargo commands inherit cwd and produce
   `target/` in unexpected places.

7. **Always** check `pgrep -af <example>` before kicking off another GPU
   training run — a stale background job pinning A100 will OOM the new
   one without an obvious error.

8. **Always** verify the example binary at `target/release/examples/<name>`
   was built with `--features cuda` before launching a GPU run.
   `cargo build -p llm-actors --release` (without the flag) silently
   overwrites the example binary with a CPU-only version, and
   subsequent runs show `device = Cpu, on_cuda = false` in the log
   and then time out on the first `generate_tokens` call after ~60s.
   Workflow: after ANY change to library code, re-run
   `CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH
   cargo build -p llm-actors --example <name> --features cuda --release`
   before relaunching. Phase 22 Stage D G1, G3, G4-agg, and the
   re-run G4-agg batches all hit this trap (Δ ~ 60 min × 5 GPUs each
   time; one G4-agg seed silently ran on CPU for ~6.5 hours before
   we noticed). **All four Phase 22 GPU binaries**
   (`phase22_he_mr_sft`, `phase22_mbpp_mr_sft`, `phase22_he_reinforce`,
   `phase22_humaneval_baseline`) now have a `PHASE22_ALLOW_CPU`
   env-var-gated fail-fast guard that bails immediately on Cpu
   rather than running for 60s+ — set `PHASE22_ALLOW_CPU=1` to
   override (for trivial smoke runs).
   **Also note:** `cargo test --workspace` builds examples too, so it
   clobbers them just like `cargo build --examples`. And checking that
   a binary's **timestamp** is unchanged does NOT tell you it was ever
   a CUDA build — an early CPU clobber leaves an old timestamp on a
   CPU binary. Verify the artifact itself:
   `strings target/release/examples/<name> | grep -c cudarc`
   (74 on a CUDA build, 0 on CPU). The Phase 22 C3-C5 session hit this
   three times.

9. **When a Pekko-driven mechanism diverges from its Python reference
   recipe, byte-compare the training step before suspecting actor
   wiring.** Phase 22 Stage D spent 4+ batches chasing over-training
   hypotheses (train-steps=100→30, gen-n=16→32→164, gen-oversample,
   etc.) before discovering the actual cause: Phase 17's
   `scripts/phase15_s1/self_improve.py:122` sets
   `labels[:prompt_ids.shape[0]] = -100` (completion-only CE loss).
   Our `train_qwen_lora_step` was computing CE on EVERY position.
   Prompts dominate (50-200 tokens vs 5-50 completion) so 80% of loss
   was prompt reproduction → catastrophic over-training, explained ALL
   regressions at once. Fix: `train_qwen_lora_step_masked` +
   `cross_entropy_with_prompt_mask` helper + `TrainSftPairs` actor
   message + `RoundConfig.sft_mask_prompt=true` default (commit
   `bc90db5`). Phase 17's second recipe diff (cosine LR schedule with
   10% warmup) shipped as commit `c7a7aed`. When porting a Python
   recipe to Pekko, START by byte-comparing the inner training loop,
   not the orchestration layer.

10. **BF16 silently corrupts `QwenModelActor` generation on LONG prompts.**
    Past ~500 total tokens, greedy/sampled output degenerates into
    token-doubling garbage after a few dozen clean tokens (`"return
    return"`, `"== =="`, `"stripstripstrip"`). BF16's 7-bit mantissa
    loses too much rotary/attention precision at length. It went
    unnoticed through all of Phase 22 because HumanEval/MBPP prompts are
    ~150 tokens with `max_new 192`; **LiveCodeBench/BigCodeBench prompts
    (500–1000+ tokens) are the first to hit it.** Isolation proof: a
    *diverse* (non-repetitive) long prompt corrupts in BF16 but is CLEAN
    in F32, temperature- and prefill-independent. Fix: run long-prompt
    generation in **F32** (`phase22_dump_completions --dtype f32`,
    default; 7B F32 = 28GB, fits a 40GB card for inference). Proper
    follow-up: F32 rotary in a vendored `candle-transformers` qwen2 to
    keep BF16 weights (found `39f038e`).

11. **Precision failures are not only a long-prompt problem — dense
    code generation breaks too, and at F16, not just BF16.** Gotcha #10
    is about BF16 past ~500 tokens. Phase 23's `PythonTool` hit the same
    class of failure with a ~40-token prompt in **F16**: an SFT'd 7B that
    reached training loss 1e-4 emitted `(p):` and `(Python(Python`
    instead of `(python print(sum(i*i for i in range(1,20+1))))`. It
    failed on **held-IN** inputs, which is what ruled out a
    generalization story and pointed at the inference path. F32 gives
    12/12 on held-out. The distinguishing feature is not prompt length
    but **completion density**: `arith` completions (~8 tokens) are
    clean at F16; python calls (~25 tokens of nested syntax) are not.
    Rule: **any example that generates code runs F32 by default**, and
    when an SFT'd model produces garbage at a loss that says it
    memorised, test held-IN before touching the recipe — a
    train/inference mismatch and an overfitting story look identical
    from the held-out number alone.

12. **The agentic loop must truncate at the tool-call boundary.** The
    model does not stop when it finishes a call: it emits the call and
    then, in the same chunk, guesses what follows — usually including a
    second copy of the call. `splice_result` preserved that tail, so the
    next step dispatched the guess as a real call (21 spurious
    dispatches over 20 arithmetic problems, and a wrong final answer
    where the duplicate polluted the buffer). `agentic_generator_actor`
    now drops everything past a newly generated call, and stop
    sequences (`with_stop_sequences`) end a chunk — **not the loop**, so
    a call inside the cut text still dispatches and still continues.
    That is what makes `"\n"` a usable stop for line-oriented formats.
    Errors went 21 → 0 and the answer rate 19/20 → 20/20.
    **The answer is tool-derived, and that is measured, not assumed**:
    `phase23_python_tool_7b --sabotage N` shifts every tool result by
    `N`. At +1 the model states the sabotaged value 12/12 and the true
    value 0/12; at +100000, 10/12 and still 0/12 (the two misses
    truncate the copy — `A: 10` for a delivered `100009` — rather than
    falling back to the true value). Run this before believing any
    "the tool computed it and the model said it" number: without it,
    tool use and independent recomputation are indistinguishable.
    It also reframes what the format SFT is for. The **base** model,
    given *unresolved* few-shot examples, already emits 12/12
    dispatchable python calls and gets 10/12 correct tool results —
    and then writes `A: 20826` for a tool that returned `17575`. Only
    3/12 of its answers track the tool, against 12/12 after SFT.
    **SFT buys grounding, not the call format.** An earlier claim here
    that the base scored 0/12 was an artifact of *resolved* shots: the
    base copies the resolved form, `parse_first_tool_call` skips it,
    and every counter reads zero. Shots must be rendered unresolved to
    measure a base model, with shot dispatches then discounted —
    `StepRecord.tool_args` exists for exactly that. And beware the
    mirror trap: on the novel families the base states the *true*
    answer 8/12 even under sabotage, because those are memorised
    Fibonacci values. Correct answers from a model holding a tool are
    not evidence the tool was used.

13. **The resolved-call marker is `→` (`tools::RESOLVED_MARKER`), not
    `=`.** Phase 4 wrote `=` into a dispatched call's body and
    `parse_first_tool_call` skips bodies containing the marker. That
    made a code-execution tool inexpressible: `=` is ubiquitous in
    source, so `(python x = 1)` read as already-resolved and was never
    dispatched. If you change the marker again, it must appear in
    neither code nor prose the model emits — and the format SFT corpora
    (`phase23_toolcall_sft` turn-2 pairs, `tool_use::render_full_trajectory`)
    embed it, so any checkpoint trained on the old marker is stale.

14. **A free verifier does not make a self-improve harvest safe, and
    self-harvest amplifies whatever it lets through.** Phase 23's
    tool-use loop verifies by executing the model's own snippet against
    a ground-truth integer — 784 completions in 6.9s, no harness to
    build. It still went wrong. Repair-derived completions were
    generated in a two-turn context and paired verbatim with the
    ORIGINAL prompt, so they carried an `A: <guess>` line ahead of the
    call. Five such pairs seeded 17%; from the next round on the loop
    harvested its own contaminated output and reached **82%** — the
    model stating an answer *before* the tool ran, which is exactly the
    property gotcha #12's `--sabotage` exists to establish. The
    verifier never sees it: the executor only ever receives the call.
    Rules: trim harvested trajectories to exactly what you intend to
    train; measure an artifact on the RAW text, because a fix that
    strips it also blinds a detector placed after the fix; and when a
    function runs on both the training and eval paths
    (`truncate_completion` does), re-score an OLD checkpoint with the
    NEW binary before believing a gain — here it confirmed 0.837 →
    0.837, so the fix moved the model and not the ruler.
    Full write-up: `docs/phase23-tooluse-self-improve.md`.

15. **Some capabilities are unreachable by sampling; the loop needs a
    channel, not more draws.** The 7B wrote `math.gcd(...)` with no
    import and failed every time — a correct algorithm against the
    tool's actual contract. **0 of 576 sampled snippets contained an
    import**, and pass@16 stayed 0/12, so a turn-1-only harvest is
    empty and the loop cannot bootstrap no matter how much you sample.
    The information existed only in the tool's error message.
    `Domain::repair_prompt` + `GeneratorActor::with_repair_failures`
    hand it back for one retry and harvest the fix only if it verifies:
    4/96 repaired against 0/24 first turns, and by the next round the
    model solved 177 first try. **The repair turn is a bootstrap
    ladder, not a crutch.** Note it does not learn what you assume —
    shown the `NameError` it dropped `math` rather than adding the
    import.

16. **A narrow harvest narrows the model; replay is free wherever
    verification is free.** Harvesting only the two unsolved families
    took them 0.000 → 1.000 and cost retention on five untouched
    families (0.988 → **0.806**, one halving) and transfer to unseen
    families (12/12 → **4/12** dispatchable calls). It over-generalised
    the idiom it had just learned — 87/160 imports where none was
    needed. Widening the harvest to all eight families restored
    retention to 1.000 and transfer to 11/12 while keeping the targets
    at 1.000, with imports **0/160** where unneeded and 49/98 where
    needed. Two riders: **saturating in a single round is a warning,
    not a success** (nothing left to learn means later rounds only
    narrow), and **scale train-steps to the corpus when you widen** —
    the rare target signal was 0.3% of the widened pool and would never
    have been sampled at the previous step count.

17. **Match the eval metric to where the training signal lives.** SFT
    self-improve **sharpens the sampling distribution**: it lifts
    aggregate pass@1 / pass@k but can *drop* the single greedy mode.
    A full-set SFT checkpoint measured 0.756 vs base 0.656 at aggregate
    pass@1 (+0.10) yet 0.439 vs 0.488 GREEDY (worse!). The first LCB
    transfer runs used greedy and wrongly concluded "no transfer"; at
    aggregate pass@1 the recipe generalizes (post-cutoff 0.041 → 0.057).
    This repo's recurring theme (pass@5 saturation, aggregate-vs-greedy):
    a "flat/negative" result must be checked against the metric where the
    gain actually lives before it is trusted.

**Benchmark scoring env (LiveCodeBench).** The official `lcb_runner` eval
core (`codegen_metrics`) runs with **`datasets`+`numpy` only** — no
torch/vllm/pyext — if you bypass the CLI wrapper (which imports torch via
`parser.py`). Isolated venv at `scratch-7b-sft/tools/lcb-venv` (built from
anaconda py3.12, which has `_bz2`; the pyenv 3.9 does not). `datasets`
must be **2.x** (the LCB dataset uses a loading script that datasets 3.x+
dropped). Generate in Rust (`phase22_dump_completions --benchmark
livecodebench`, F32), score with `scripts/phase22_bench/lcb_score.py`.

**The measured 7B self-improve recipe.** Don't re-derive these; they cost
~2 weeks of A100 time. Defaults are already set in
`scripts/phase22_rl_variance/arm_sweep.sh`.

```
--advantage-mode mean --pg-positive-only --k-per-prompt 16 --rl-steps 30
```

- **`--k-per-prompt 16` is the saturation point**, measured on LiveCodeBench
  post-cutoff transfer (6 seeds/arm): K=2 0.084 / K=4 0.098 / K=8 0.111 /
  **K=16 0.129** / K=32 0.128. Log-linear to 16 (+0.0148 per doubling, 6/6
  seeds, t=3.68), flat from 16 to 32 (−0.0014, t=−0.17). K=32 doubles the
  harvest cost for nothing **and** trains better in-domain while transferring
  no better — past saturation the extra harvest buys hard-tail fit that does
  not generalise.
- **The operating point depends on the objective.** Transfer saturates at
  K=16, but *in-domain* keeps improving to K=32 — and its variance collapses
  there. In-domain hard-tail pass@1 (6 seeds, same ruler): K=2 0.298±0.098 /
  K=4 0.399±0.090 / K=8 0.538±0.076 / K=16 0.572±0.069 / **K=32 0.606±0.037**.
  σ is monotone across all five (p≈0.008 under random ordering) and K=32 lands
  on SFT's σ (0.037) with 1.7× SFT's mean. Post hoc — no pairwise step is
  significant at n=6 (K=8 vs K=32: F=4.34, p≈0.13) — so **use K=32 when
  in-domain reliability is what you want, K=16 when transfer is**, and confirm
  with more seeds before treating the variance result as settled.
- **The harvest gain is LiveCodeBench-specific.** On BigCodeBench Hard
  (difficulty/realism axis, same 6 seeds, same ruler) K=16 is **flat** vs K=8:
  paired −0.0018 (t=−0.19, 2/6), where LCB gave +0.0188 (t=3.58, 6/6). K=16
  still stays the default because it wins on LCB, loses nothing on BCB, and
  **halves the spread there** (σ 0.0095 vs K=8's 0.0163) while beating SFT in
  6/6 seeds. Benchmark-axis dependence — already known for RL-vs-SFT (+4σ on
  LCB, +1σ on BCB) — holds for the harvest lever too. Don't assume a harvest
  gain measured on one axis shows up on another.
- **`--pg-positive-only` matters in-domain, not for transfer.** Bounding the
  objective is worth **+0.2070 pass@1 in-domain (12/12 seeds, t=6.67,
  p<0.0001)** — *established* by pre-registered replication on fresh seeds
  (`docs/phase22-c1-prereg.md`), not the earlier optional-stopping estimate of
  +0.124. It is null out-of-domain (t=−0.52), so expect the benefit in-domain
  only. Keep it on.
- **RL vs SFT**: same mean in-domain with ~3× the variance (so SFT is the
  in-domain deployment pick), but RL transfers ~2–6× better to unseen
  problems. Pick by which one you need.
- **Only the trend is significant, not the steps.** K=16−K=8 clears noise on
  its own (t=3.58); K=4−K=2 and K=8−K=4 do not (t≈1.5). A pairwise-only
  reading would have concluded "no effect" three times. Budget for the whole
  sweep, not two points.
- Wall clock per step, 64 prompts: K=4 ~15 min, K=8 ~36, K=16 ~70, K=32 ~78.
  Generation does **not** scale linearly with K — fixed per-step cost
  dominates at large K.

Full write-ups: `docs/phase22-c4-c5-rl-vs-sft.md`,
`docs/phase22-livecodebench-notes.md`.

## Testing strategy

- 74 unit tests are exhaustive on what's deterministic (parsing,
  domain verification, evolution operators, EWC penalty math, LoRA
  output shape, RustCodeDomain dispatch, cfg.json round-trip,
  ensemble validation + smoke).
- Integration validation is via the worked-out examples in
  `nanogpt-rs/examples/` and `llm-actors/examples/`. Each example
  asserts something concrete (correct count, non-zero gradient, etc.).
- CI lives at `.github/workflows/ci.yml` — `cargo build --workspace
  --release`, `cargo test --workspace --release`, `cargo build
  --workspace --examples`, `cargo fmt --check`, and
  `cargo clippy --workspace --all-targets -- -D warnings`. All four
  gates are strict (no `continue-on-error`).
- Workspace is currently at **zero clippy warnings** and **zero fmt
  drift**. Don't introduce them — fix the lint or, if the lint is
  genuinely the wrong call, gate the function with
  `#[allow(clippy::xxx)]` and a comment explaining why.

## Pre-commit checklist

Before committing, run these in order. If any fails, fix before
committing — do not let CI catch it:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Optional but useful for example-touching changes:

```bash
cargo build --workspace --examples --release
```

## Skills (Claude-specific)

Project skills live in `.claude/skills/`; they are auto-discovered every
session. **`rust-guardrails` is the default skill for Rust work here** — invoke
`/rust-guardrails` (or apply it as a checklist) whenever you:

- add or review a type that implements a trait by **wrapping** another impl of
  that trait (a new `Domain`/actor wrapper). A forgotten *defaulted*-method
  delegation is a silent-failure surface the compiler can't catch — it is what
  disabled `truncate_completion` for the whole hard-tail series. Use
  `assert_domain_fully_delegates!` (`llm-actors/src/domain/delegation_probe.rs`)
  and, for pure pass-throughs, the `ambassador` `#[delegate]` macro.
- add or change an **eval/measurement** path. Compare the base measurement
  against the published number via `llm_actors::eval_sanity` in the comparable
  config (full-set, greedy, unfiltered, no checkpoint); never compare a
  filtered/subset number to a public/unfiltered baseline. `phase22_humaneval_baseline
  --sanity-strict` fails CI on drift.

The skill encodes the C4/C5 lessons (`docs/phase22-c4-c5-rl-vs-sft.md` #1/#6).

## Memory system (Claude-specific)

The user has an auto-memory at
`~/.claude/projects/-raid-users-paul-workLLM/memory/`. The 14 files
there capture phase-by-phase decisions and results — read them when
ramping up to understand WHY a particular choice was made, not just
what was implemented. Key entries:

- `MEMORY.md` — the index
- `phase3_*` — NAS turns, especially `phase3_norm_status.md` (the
  final turn that hit fitness 0.49)
- `phase4_*` — agent loop + forgetting comparisons
- `phase1_kowiki_status.md` — Korean training validation

When making changes, update memory entries in the same turn. Don't
rely on git history alone for "why" — git tells you what changed,
memory tells you what was learned.

## When something is broken

Order to check:
1. **Build green?** `cargo clippy --workspace --all-targets -- -D
   warnings` and `cargo fmt --check` should both pass — they're CI
   gates. If they don't, that's the first thing to fix.
2. Did `nano_50m` (or whichever preset) silently change in `config.rs`?
   If old checkpoints fail to load, this is almost always why.
3. Is the right CUDA toolchain in PATH? See gotcha #5.
4. Did a previous example leak GPU memory (`nvidia-smi` says it's full
   but no `python`/`example` process is visible)? `pkill -f train_` and
   wait 30s.
5. Is the corpus actually clean? `head -200 data/kowiki/kowiki_clean.txt`
   should show prose, not `[[파일:...]]` or LaTeX-only lines. If not,
   `extract_kowiki` regressed; check the line-level filters at the top
   of `clean()`.

## Large checkpoints + Git LFS

`*.safetensors`, `*.bz2`, `*.bin`, `*.pt`, `*.gguf` are routed through
Git LFS via `.gitattributes`. Today the `checkpoints/` and `data/`
trees are gitignored, so LFS is dormant — but if you ever want to ship
a "reference" checkpoint with the repo (e.g. the 30K-step KoWiki run
backing the README results table), see `docs/lfs-setup.md`. Do **not**
commit the raw `kowiki-latest-pages-articles.xml.bz2` (~1.2 GB) — it
blows the GitHub free LFS tier and is reproducibly downloadable from
`dumps.wikimedia.org/kowiki/latest/`.
