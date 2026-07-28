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
