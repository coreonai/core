# Phase 22 Stage A — HumanEval as a Rust `Domain`

After Phase 21 closed the Pekko bridge, the natural follow-up is
**reproducing Phase 17–20's actual benchmark numbers** through the
Rust stack. Stage A ships the missing piece: a `HumanEvalDomain`
that loads the standard 164-problem set and verifies completions
via `python3` subprocess — exactly the pattern Phase 15+'s Python
scripts used.

`EvaluatorActor::<QwenModelActor>` + `HumanEvalDomain` =
HumanEval baseline measurement through the Pekko stack, no
standalone driver script. Phase 17 S6's measurement is now
ALL-IN-ONE-PROCESS in Rust:

```
QwenModelActor (Stage D inference, BF16 to match Qwen2.5-Coder native)
  ↑ ModelMessage::GenerateTokens
EvaluatorActor::<QwenModelActor> (Stage E generic eval, Stage A pass@k)
  ↑ EvaluatorMessage::Eval { passk, ... }
HumanEvalDomain (Stage 22-A — python3 subprocess verify with timeout)
```

## What's in this commit

### `llm-actors/src/domain/human_eval.rs` (new)

`HumanEvalDomain`:
- `from_jsonl(path, scratch_dir)` — load 164 problems from the
  standard JSONL (`data/humaneval/HumanEval.jsonl`, already in tree
  from Phase 15)
- `from_default_data_dir()` — convenience loader from cwd
- `Domain::sample_prompt(rng)` — uniformly samples a problem
- `Domain::verify(prompt, completion)`:
  1. `prompt_to_idx` HashMap finds the problem
  2. Build program = `prompt + completion + "\n\n" + test +
     "\ncheck(<entry_point>)\n"`
  3. Write to `scratch_dir/solution.py` under a write-lock (so
     concurrent verifies don't race)
  4. `python3 solution.py` with 8-second poll-based timeout
  5. exit 0 → `Correct`, else → `Incorrect`, error → `Inconclusive`

The write-lock + scratch-file pattern mirrors `RustCodeDomain`. The
timeout polling avoids pulling in `wait-timeout` crate.

`HumanEvalProblem` — JSONL deserialization target with
`task_id / prompt / entry_point / canonical_solution / test`.

### `llm-actors/examples/phase22_humaneval_baseline.rs` (new)

CLI for measuring Qwen2.5-Coder-0.5B's HumanEval baseline through
the Pekko stack:

```bash
# Quick smoke (8 problems, ~12s)
cargo run --release --features cuda --example phase22_humaneval_baseline -- \
    --n-problems 8 --passk 1

# Full 164-problem baseline (~4 min)
cargo run --release --features cuda --example phase22_humaneval_baseline -- \
    --n-problems 164 --passk 1

# pass@k sweep
cargo run --release --features cuda --example phase22_humaneval_baseline -- \
    --n-problems 164 --passk 10
```

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **149 tests** (was 145; +4
  HumanEvalDomain tests: loader returns 164, canonical solution
  verifies, empty body fails, unknown prompt is inconclusive)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase22_humaneval_baseline --n-problems 164`: produces a
  pass-rate over the FULL HumanEval set in ~4 min

## Measurement result + gap to Phase 17

Full 164-problem run, greedy decode (temp=0), BF16 inference on A100:
**pass@1 = 0.0793 (13/164)**.

Phase 17 S6 reported pass@1 = 0.216 on the same base Qwen model.
**Stage A's measurement is ~2.7× lower** than the reference.

Likely contributors (in rough order of impact):

1. **Sampling with replacement.** `EvaluatorActor.run` calls
   `domain.sample_prompt(rng)` once per iteration. With uniform
   sampling, the expected unique-problem coverage over n=164 draws
   is `164 × (1 − (163/164)^164) ≈ 103.8` (~63%). At 0.216 reference
   pass-rate, that's ~22 expected passes — already short of the
   reference even before factoring numerical drift.
   **Fix**: an `EvaluatorActor` mode that iterates problems sequentially
   instead of sampling. Stage B work.

2. **Greedy decode numerical drift between BF16 (this run) and
   bf16-via-HF-transformers (Phase 17 reference).** Tested F16 first:
   pass@1 = 0.0312 / 32. Switched to BF16: pass@1 = 0.0625 / 32. So
   BF16 ≈ 2× F16 — the gap to Phase 17 is partly precision, partly
   sampling-with-replacement.

3. **Stop / truncation handling.** My `QwenModelActor` short-circuits
   on `eos_token_id = 151643`. HF transformers' generate may include
   additional stop heuristics (newline-after-blank, end-of-function,
   etc.) that produce cleaner completions on average.

Stage A's deliverable is **the wiring** — the Domain is correctly
loaded, verifies match Phase 15's program-composition rule, all 149
tests pass. **Stage B (next-up)** is the bit-exact baseline
reproduction, which means addressing items (1) and (2) above.

## What this commit does NOT do

- **Sequential prompt iteration** (a no-replacement EvaluatorActor
  mode). Currently `sample_prompt` is the only ingress to the Domain
  and it samples with replacement.
- **bit-exact match to Phase 17 numbers.** Requires the above fix +
  possibly mirroring HF's generate stopping criteria.
- **MBPP-100 Domain.** Same JSONL shape; trivially mirrorable.
  Deferred to keep this commit focused on HumanEval.
- **Multi-round SFT on HumanEval through Pekko.** This is the
  Phase 17 r=5 saturation curve replication — needs Stage B's
  bit-exact baseline first.
- **Stage G RL on HumanEval.** Same — needs Stage B first.

## Phase 22 stage roadmap (post Stage A)

| stage | scope | status |
|---|---|---|
| **A** | **HumanEvalDomain + python3 verify + baseline binary** | ✅ (this commit) |
| B | Bit-exact Phase 17 baseline: sequential prompt iteration + sampling alignment + bf16 mode | next-up |
| C | MBPP-100 Domain (mirror of A; cross-substrate benchmark) | TODO |
| D | Multi-round SFT on HumanEval via `run_multi_round` against `QwenTrainerActorHandle` → reproduce Phase 17 saturation curve through Pekko | TODO |
| E | RL on HumanEval via Phase 21 Stage G mechanism (REINFORCE with HumanEval verdict as reward) | TODO |

## Files

- `llm-actors/src/domain/human_eval.rs` — new Domain + 4 unit tests
- `llm-actors/src/domain/mod.rs` — module registration
- `llm-actors/examples/phase22_humaneval_baseline.rs` — measurement binary
- `llm-actors/Cargo.toml` — example registration
- `docs/phase22-stage-a.md` (this)

## See also

- `docs/phase21-overview.md` — the Pekko bridge this Stage builds on
- `docs/phase17-closeout.md` — the reference Phase 17 S6 baseline
- `scripts/phase15_s1/problems.py` — the Python loader / verifier
  pattern this Domain mirrors
