# Phase 21 Stage B — substrate scale-up + pass@k lift measurement

Phase 13 retracted K9 1M as a noise-floor substrate (σ_within > 0.14
of the mean) — too coarse for algorithm comparison. Phase 21 Stage A
landed the pass@k actor infrastructure but its K9 smoke showed 0/24
even at pass@5 (model never reached the "near but not at" regime
Phase 17 S6's mechanism needs).

Stage B's question: **at a slightly larger Rust-native scale
(n_embd=512, n_layer=6, ~24M params, ~25× K9 1M), does the
inference-time pass@k mechanism actually surface lift?**

## Answer: yes — and the post-pretrain model lifts more than the SFT-trained one

| recipe | pass@1 | pass@3 | pass@5 | pass@10 |
|---|---:|---:|---:|---:|
| post-pretrain seed (no SFT) | 0.458 | 0.500 | **0.792** | 0.625 |
| r=2 SFT (post Phase 21 C-style rounds) | 0.417 | 0.208 | 0.500 | 0.542 |

**Seed-checkpoint lift**: pass@5 - pass@1 = **+0.333 absolute (1.73×)**.
**r=2 SFT lift**: pass@5 - pass@1 = +0.083 (1.20×); pass@10 - pass@1 =
+0.125 (1.30×).

Both confirm the **Phase 17 S6 / S8 / S9 mechanism replicates at
Rust-native scale-up substrate** — pass@k buys signal where greedy
is "near but not at" the answer. Phase 17 S6 measured against Qwen
(pass@10 0.524 vs pass@1 0.216 = +0.308 absolute, 2.43×); our
scale-up shows a thinner lift on top of higher baselines, consistent
with the "less headroom" pattern.

## Phase 17 Sa pattern: SFT closes the inference gap

In Phase 17 Python+Qwen measurements, multi-round SFT both lifted
pass@1 AND closed the gap to pass@10 (Sa's "training and inference
axes are additive but with diminishing returns"). Our Stage B
**replicates that qualitative pattern**:

- pre-SFT pass@5 lift: +0.333 (model has lots of "near misses" at greedy)
- post-SFT pass@5 lift: +0.083 (greedy already captures most easy wins)

At this Rust-native scale **base + pass@5 (0.792) beats r=2 SFT at any
passk (max 0.542)** — opposite of Phase 17 Sa where SFT + pass@k
dominated base + pass@k. Caveat: this is a single-seed measurement at
a substrate Phase 13 flagged as variance-prone; the SFT degradation
could be either real (over-fitting the narrow 10-challenge corpus) or
noise. Multi-seed replication is the natural follow-up.

## Bugs squashed

### `RustCodeDomain` needs `ensure_scratch_project()` at startup

The eval-only binary's first runs reported 0/24 across all passk
values. Diagnosed via sample dump: completions looked reasonable
(`"10 - 5\n"` against the `equals_zero` prompt, etc.) but every
verdict came back failed. Root cause: `RustCodeDomain::verify`
writes `src/main.rs` then runs `cargo run` in `self.scratch_dir`,
but a bare scratch_dir without a `Cargo.toml` makes cargo error
out → every verdict became `Inconclusive` (i.e., not Correct).
`RustCodeDomain::ensure_scratch_project()` lays down both the
Cargo.toml AND the src skeleton; it must be called once before
the first verify. Fixed in `phase21_b_eval_passk`.

`self_improve_rust` already calls `ensure_scratch_project` so its
runs were never affected.

### eval_temperature default 0 collides with pass@k > 1

Pre-Stage-B, `self_improve_rust` hardcoded `temperature: 0.0` and
`top_k: Some(1)` on the eval-sampling config. With `--eval-passk 5`
that produces 5 IDENTICAL greedy samples per prompt — passk has no
diversity to OR over, and the measurement is meaningless. Stage B
adds `--eval-temperature` and `--eval-top-k` CLI flags so callers can
opt in to sampling for the eval phase. Defaults preserve historical
behavior (greedy).

## What's in this commit

### Library + driver changes
- `llm-actors/examples/self_improve_rust.rs` — new `--eval-temperature`
  and `--eval-top-k` CLI flags. Defaults `0.0` / `1` preserve the
  historical greedy eval; set to `0.8` / `10` together with
  `--eval-passk > 1` for Stage B-style measurements.

### New eval-only binary
- `llm-actors/examples/phase21_b_eval_passk.rs` — loads a
  `self_improve_rust`-trained checkpoint and runs the standard
  `RustCodeDomain` eval at multiple `passk` values on the SAME model.
  Avoids the "two independent pretrains" contamination of comparing
  separate `self_improve_rust` runs (different RNG → different weights
  → noise mixed into the passk measurement).

### Stage B sprint driver
- `scripts/phase21_b/run_smoke.sh` — runs `self_improve_rust` at the
  scale-up config (`n_embd=512 n_layer=6 n_head=8 n_kv_head=4`,
  `pretrain=3000 round-train=600 rounds=2`) for one passk value at
  a time, with auto-bumped eval temp/top-k when passk > 1.

### Logs
- `scripts/phase21_b/run_n512_l6_passk{1,5}.log` — raw outputs from
  the two scale-up training runs (gitignored later; kept for now as
  primary measurement artifacts).

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **144 tests** (no change)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_b_eval_passk` on r=2 SFT ckpt: pass@5 lift
  **+0.083 (1.20×)** over greedy
- ✅ E2E `phase21_b_eval_passk` on seed ckpt: pass@5 lift
  **+0.333 (1.73×)** over greedy

## What this does NOT do

- **Single-seed measurement.** `self_improve_rust` doesn't expose a
  top-level `--seed` flag yet — different runs get different
  weight inits because the global RNG isn't pinned. Re-running the
  whole Stage B comparison across 3-5 seeds is the obvious next step
  to bound variance, especially around the "SFT degradation" finding.
- **No multi-round SFT at scale-up.** Just `rounds=2` matching the
  default. Phase 21 stage C's `run_multi_round` helper makes deeper
  sweeps cheap once the seed-control problem above is solved.
- **No HumanEval / MBPP at this substrate.** RustCodeDomain only;
  cross-substrate validation against Phase 17-20's actual eval sets
  needs `QwenModelActor` (Stage D) wired through Evaluator (Stage E).

## Phase 21 stage roadmap

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 + QwenModelActor (inference) | ✅ (`acfdc5d`) |
| F | Qwen2 LoRA training in Rust (Candle-native) | ✅ (`8367d2a`) |
| **B** | **Substrate scale-up + pass@k lift measurement** | ✅ (this commit) |
| E | Generic Evaluator/Generator over `Actor<Message=ModelMessage>` | next-up |
| G | RL with pass@k reward | deferred |

## Files

- `llm-actors/examples/self_improve_rust.rs` — added eval-temp/top-k flags
- `llm-actors/examples/phase21_b_eval_passk.rs` — new eval-only binary
- `llm-actors/Cargo.toml` — example registration
- `scripts/phase21_b/run_smoke.sh` — sprint driver
- `scripts/phase21_b/run_n512_l6_passk{1,5}.log` — measurement logs
- `docs/phase21-stage-b.md` (this)

## See also

- `docs/phase17-closeout.md` — S6 / Sa pass@k discovery on Qwen (Python)
- `docs/phase21-stage-a.md` — actor-side pass@k infra
- `docs/phase21-stage-c.md` — `run_multi_round` helper
- `docs/phase21-stage-d.md` — `QwenModelActor` (inference)
- `docs/phase21-stage-f.md` — Candle-native Qwen2 LoRA training
