---
name: rust-guardrails
description: >-
  Apply this repo's two hard-won Rust safety disciplines whenever you add or
  review Rust that (a) implements a trait by WRAPPING another impl of that
  trait (e.g. a new `Domain`/actor wrapper like `FilteredDomain`), or (b)
  adds/changes an eval or benchmark MEASUREMENT path. Both are silent-failure
  surfaces the compiler cannot catch — a forgotten defaulted-method delegation
  and a mis-measured metric each cost the Phase 22 study batches of GPU time
  and inverted a conclusion. Invoke before writing such code, and as a review
  checklist after.
---

# rust-guardrails

Two bug classes in this repo are invisible to the compiler and have each caused
a wrong headline. This skill is the checklist that catches them. Background:
`docs/phase22-c4-c5-rl-vs-sft.md` (Lesson #1 and #6) and CLAUDE.md gotcha #8/#9.

## Guard 1 — Trait-wrapper delegation completeness

**The trap.** A type that implements a trait by holding another impl and
forwarding calls will *silently inherit the trait's default* for any method it
forgets to delegate — no error, no warning. `FilteredDomain` inherited
`Domain::truncate_completion`'s identity default, which switched completion
truncation OFF for every hard-tail experiment and mis-measured the 7B base
(0.246 vs the true 0.422), inflating every reported gain built on it.

**Do this for any wrapper that implements a trait by delegation:**

1. **Enumerate every trait method, including defaulted ones.** `Domain`
   (`llm-actors/src/domain/mod.rs`) has 3 required + 4 defaulted (`score`,
   `n_prompts`, `nth_prompt`, `truncate_completion`). The defaulted ones are
   the danger — required methods won't compile if missing.
2. **Every method is either delegated or intentionally overridden with a
   `// ` comment saying why.** No method should reach a wrapper by trait
   default. A transformed method (e.g. `FilteredDomain` re-indexes
   `nth_prompt`) is fine — an accidentally-defaulted one is the bug.
3. **Add the delegation test.** For `Domain` wrappers, one line:
   ```rust
   use crate::domain::delegation_probe::assert_domain_fully_delegates;
   assert_domain_fully_delegates!(|inner| MyWrapper::new(inner, ...));
   ```
   `ProbeDomain` returns a non-default sentinel from every defaulted method, so
   the macro fails if the wrapper falls back to any trait default. See
   `llm-actors/src/domain/delegation_probe.rs` and the retrofitted test in
   `filtered.rs` (`all_defaulted_methods_delegate`).
4. **Prefer compile-time delegation for PURE pass-throughs.** If a wrapper
   forwards *every* method unchanged (no re-indexing/transform), use the
   `ambassador` crate: `#[derive(Delegate)] #[delegate(Trait, target="inner")]`.
   That makes the compiler enforce completeness. Add `ambassador` to
   `Cargo.toml` when the first such wrapper appears (none today —
   `FilteredDomain` selectively overrides, so it uses the test guard instead).
5. **For a new trait of your own with defaulted methods that wrappers must
   delegate:** write an equivalent `assert_*_fully_delegates!` probe next to it.
   A defaulted trait method is a silent-failure surface by construction.

## Guard 2 — Public-baseline sanity in eval/measurement paths

**The trap.** An internal metric that is never checked against a known-good
reference can be badly wrong and still look like a result. The C5 reward bug
and the FilteredDomain truncation bug both produced numbers that were ~2× off
and were compared to other numbers anyway.

**Do this for any new/changed benchmark or eval path:**

1. **Compare against the published baseline in the comparable config.** For
   Qwen HumanEval/MBPP, use `llm_actors::eval_sanity::check_public_baseline`
   (official base-model greedy full-set pass@1 from arXiv:2409.12186 Table 5:
   7B HumanEval 0.616, 0.5B 0.280, MBPP 0.769/0.529). Print a `[SANITY]` line;
   gate CI with a strict flag (`phase22_humaneval_baseline --sanity-strict`
   exits non-zero on drift).
2. **Comparability is the whole point.** Only a like-for-like run is
   comparable: full set, greedy (`passk 1`), unfiltered domain, no trained
   checkpoint. A filtered/subset/sampled number is NOT comparable to a public
   or unfiltered baseline — surface that explicitly (the eval binary prints
   `[SANITY] WARN filtered ...`). Never compare across measurement paths
   without re-measuring both on the same ruler.
3. **When a comparison to a prior/published number fails a sanity check,
   re-measure the prior number** rather than explaining the gap away — the
   0.246 base was a 1.75× inflation that a sanity check would have caught at
   measurement time.
4. **Add the sanity numbers to `eval_sanity.rs`** when you introduce a new
   model or benchmark, citing the source, with a tolerance wide enough for
   harness differences (~0.10) but tight enough to catch a ~2× miss.

## GPU-build hygiene (carry-over, gotcha #8)

If your change touches library code that an example depends on, the running
example binaries are stale until rebuilt **with `--features cuda`**. `cargo
test`/`cargo build --examples` (no cuda) silently overwrite the CUDA example
with a CPU one. Verify with `strings target/release/examples/<name> | grep -c
cudarc` (74 = CUDA, 0 = CPU). When a GPU job is mid-flight, avoid
`--release --examples` rebuilds until its checkpoints are evaluated; use
`cargo test -p <crate> --lib` and dev-profile clippy, which don't touch the
release example binaries.

## Before committing

Run the repo's checklist: `cargo fmt --all` → `cargo clippy --workspace
--all-targets -- -D warnings` → `cargo test --workspace`. The delegation macro
and `eval_sanity` tests run as part of it.
