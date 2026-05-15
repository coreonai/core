# Phase 21 Stage C — `run_multi_round` helper API

Phase 21 Stage A wired inference-time pass@k into `llm-actors`. Stage C
adds the **multi-round orchestration helper** that Phase 17-20's
Python scripts had inline but the Rust actor stack did not — every
example previously hand-rolled its own `for round in 0..rounds` loop
with manual `init_from / save_path` chaining and per-round seed
bumps.

`run_multi_round` centralizes that pattern, derives Clone on
`RoundConfig`, and ships with a tiny smoke example
(`phase21_multi_round_smoke`) that exercises the helper end-to-end on
`ArithmeticDomain`.

## API surface

```rust
pub use llm_actors::{
    run_multi_round,      // pub async fn
    MultiRoundConfig,     // pub struct (Clone)
    RoundConfig,          // existing — now also Clone
    RoundActors,          // existing
};

pub struct MultiRoundConfig {
    pub rounds: usize,
    pub base: RoundConfig,      // template; per-round mutations applied
    pub gen_seed_stride: u64,   // default 1 via MultiRoundConfig::new
}

pub async fn run_multi_round<F>(
    actors: &RoundActors,
    cfg: MultiRoundConfig,
    mut on_round_done: F,
) -> anyhow::Result<Vec<RoundReport>>
where F: FnMut(usize, &RoundReport)
```

Per-round mutations applied automatically:

| field | mutation |
|---|---|
| `round` | current round index `r` |
| `init_from` | previous round's `save_path` (round 0 uses `base.init_from`) |
| `save_path` | `<base.save_path stripped of .safetensors>.r{r}.safetensors` |
| `gen_seed` | `base.gen_seed + r * gen_seed_stride` |
| `gen_sampling.seed` | `Some(base.gen_sampling.seed.unwrap_or(0) + r)` |
| `corpus_seed` | `base.corpus_seed.map(|s| s + r)` |

All other fields (eval_seed/sampling, train_cfg, anchor, dpo_*,
freeze_base, eval_passk, gen_oversample, sample_mode, min_corpus_chars)
are held constant.

## What's in this commit

### Library changes
- `RoundConfig` gains `#[derive(Clone)]` (all fields already Clone).
- `MultiRoundConfig` + `run_multi_round` + `save_path_template` helper
  added to `supervisor.rs`.
- `lib.rs` re-exports `run_multi_round` and `MultiRoundConfig`
  alongside the existing `run_round` / `RoundConfig` / `RoundActors`.

### Tests
4 new unit tests in `supervisor::tests`:
- `save_path_template_strips_safetensors_suffix`
- `save_path_template_leaves_paths_without_suffix`
- `save_path_template_strips_only_trailing_safetensors`
- `multi_round_config_new_defaults_stride_to_one`
- `round_config_is_clone` (compile-time Clone assertion on both
  `RoundConfig` and `MultiRoundConfig`)

Total: **141 tests** pass (88 in llm-actors + 53 in nanogpt-rs).

### New example
`llm-actors/examples/phase21_multi_round_smoke.rs` — minimal
end-to-end exerciser:
1. Build char-tokenizer + tiny GPT (2 layer, 2 head, 64 embd) on
   `ArithmeticDomain`.
2. Brief pretrain (200 steps).
3. Spawn all actors.
4. Call `run_multi_round` with `rounds=3, eval_passk=3, gen_n=40`
5. Print per-round + summary; assert `reports.len() == 3`.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **141 tests** (88 + 53)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E: `./target/release/examples/phase21_multi_round_smoke` prints
  `phase21_multi_round_smoke: PASS` after 3 rounds with init_from
  chain `r0 → r1 → r2`, `EvaluatorActor done passk=3` in logs

## Why pass rate stays at 0

Same reason as Stage A's K9 smoke: a tiny char-GPT with 200 pretrain
steps cannot learn even simple addition. The smoke is **wiring
verification**, not signal measurement. The acceptance criterion is
that `run_multi_round` produces 3 chained reports — not that the
arithmetic eval passes.

The training-loss trajectory (~2.30 at round 2) confirms training is
actually happening; it's just nowhere near convergence for arithmetic
at this scale. Phase 13 retracted K9 1M as a noise-floor substrate,
and that finding applies to this even smaller demo.

## Migration note for existing examples

Existing examples (`self_improve_round`, `self_improve_rust`,
`self_improve_korean`, `self_improve_ensemble_rust`) deliberately
**not migrated**:

- Each has hand-tuned per-round seed schemes (e.g., `corpus_seed:
  Some(round * 31 + 7)`) that the helper's uniform `+round` doesn't
  reproduce.
- Each prints round-specific diagnostics tied to a CLI/config local
  to the example.
- Migration would be net-loss for reproducibility of past results.

New code should use `run_multi_round`; old code stays as it is.

## Phase 21 Stage roadmap (post Stage C)

| stage | scope | status |
|---|---|---|
| A | Pass@k in actor stack | ✅ (`7a5d18b`) |
| **C** | **`run_multi_round` helper + smoke** | ✅ (this commit) |
| B | Substrate scale-up (n_embd=512, n_layer=6) + measure passk lift | deferred — needs ~5-10× K9 wallclock to surface signal |
| D | HF Qwen `ModelActor` impl (Pekko ↔ Python bridge) | days of infra |
| E | RL with pass@k reward | days of code |

Note: Stage C originally listed "smoke at scale-up" — that's been
split out as Stage B since the helper + scale-up smoke are
independent. The helper is shippable now; scale-up smoke is its
own measurement run.

## Files

- `llm-actors/src/supervisor.rs` — `MultiRoundConfig`, `run_multi_round`,
  `#[derive(Clone)]` on RoundConfig, 5 unit tests
- `llm-actors/src/lib.rs` — re-exports
- `llm-actors/examples/phase21_multi_round_smoke.rs` — new smoke
- `docs/phase21-stage-c.md` (this)

## See also

- `docs/phase21-stage-a.md` — pass@k integration (Stage A)
- `docs/phase20-closeout.md` — saturation curve + deployment recipe
  (the Python findings this helper now mirrors in Rust)
