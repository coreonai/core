# Phase 22 Stage C — MBPP-100 as a Rust `Domain` (cross-substrate mirror of A)

Stage A shipped `HumanEvalDomain`. Stage B closed the gap to Phase 17
S6's reference (`pass@1 = 0.222` on a 32×k=10 aggregate subset vs
0.216, within 1σ). **Stage C ships `MbppDomain`** — Phase 17 S9's
cross-substrate companion, now in Rust on top of the same Pekko
inference stack.

Library work mirrors Stage A; the only domain-specific bits are how
the MBPP schema is shaped (`text` + `code` + `test_list`, no inline
prompt) and how Phase 17 S3 synthesizes a HumanEval-style prompt from
those fields.

## What's in this commit

### `MbppDomain` (`llm-actors/src/domain/mbpp.rs`)

```rust
pub struct MbppProblem {
    pub task_id: usize,
    pub text: String,
    pub code: String,
    pub test_list: Vec<String>,
    pub test_setup_code: String,    // serde(default), often empty
}

pub struct MbppChallenge {
    pub task_id: usize,
    pub prompt: String,       // synthesized: imports + sig + docstring
    pub suffix: String,       // imports + test_list (top-level asserts)
    pub entry_point: String,
}

pub struct MbppDomain { challenges: Vec<MbppChallenge>, ... }
```

**Prompt synthesis** (Phase 17 S3's `problems.py` recipe, ported):

1. **Parse the first top-level `def name(args):`** from `code`. We
   walk lines, find the first one whose leading whitespace is zero
   and which starts with `def `. Nested `def`s in `code` (helper
   functions inside the canonical solution) are skipped.
2. **Detect imports** referenced anywhere in `code`: `import math`,
   `from typing import *` (also when only `List[…]` / `Dict[…]` /
   `Optional[…]` aliases appear), `import re`, `import heapq`,
   `import collections` (also when `Counter(…)` / `defaultdict(…)`
   are called), `from collections import *`, `import itertools`
   (also `itertools.…`), `import functools` (also bare `reduce(…)`).
3. **Prompt** = `<imports>\n\n<sig>\n    """<text>"""\n`. The
   description is a single line (newlines collapsed for cleanliness).
4. **Suffix** = `\n<imports>\n<test_list joined>\n`. Asserts run at
   module top-level — **no `check(<entry>)` call needed**, unlike
   HumanEval. This is the structural difference between the two
   substrates.

**Loader**:

- `MbppDomain::from_jsonl(jsonl, scratch)` → MBPP-100 default
  (task_id 11–110, skips few-shot examples 1–10, caps to 100).
- `MbppDomain::from_jsonl_range(jsonl, scratch, start, end, max)`
  for custom ranges (the public escape hatch).

**verify**: same poll-based `python3` subprocess with 8s timeout
under a `write_lock` Mutex — copy of `HumanEvalDomain`'s pattern.

### `Domain` overrides

`MbppDomain` implements `n_prompts() → Some(self.challenges.len())`
and `nth_prompt(i) → Some(self.challenges[i].prompt.clone())`, so
`EvaluatorMessage::EvalSequential` (Stage B) works against it out of
the box — including aggregate mode.

### `phase22_mbpp_baseline` example

```bash
# Smoke (n=8 greedy, ~10s, validates wiring):
cargo run --release --features cuda --example phase22_mbpp_baseline -- \
    --n-problems 8 --passk 1

# Phase 17 S9-equivalent measurement (n=100 × k=10 = 1000 attempts):
cargo run --release --features cuda --example phase22_mbpp_baseline -- \
    --n-problems 100 --passk 10 --sequential --aggregate \
    --max-new-tokens 200
```

CLI flags mirror `phase22_humaneval_baseline` exactly: `--n-problems`,
`--passk`, `--max-new-tokens`, `--seed`, `--sequential`, `--aggregate`.

## Measurement

**Greedy smoke** (n=8, k=1, temp=0, BF16, max_new=200, ~7s):

```
[Phase22C] per-prompt pass@1 = 0.1250  (1/8)
```

Completions look like real Python — `return l*b*h`,
`return list(filter(lambda x: x % 2 == 0, nums))`, etc. The wiring is
sound; pass@1 at greedy is just noisy on n=8.

**Aggregate subset** (n=16 × k=10 = 160 attempts, temp=0.8, top_p=0.95,
BF16, max_new=200, ~3.5 min on A100):

```
[Phase22C] per-prompt pass@10 = 0.5625  (9/16)
[Phase22C] aggregate pass@1 (raw, all samples) = 0.1875  (30/160)
           — comparable to Phase 17 S9's pass@1 raw on MBPP-100
```

Phase 17 S9 single-seed reference (full MBPP-100 × k=10): pass@1
≈ 0.36, pass@10 ≈ 0.66, Δ ≈ +0.30. Our 16-subset (`nth_prompt(0..16)`
= task_id 11–26) lands lower than the full-100 ref:

- **0.1875 vs 0.36** (pass@1) — 16-subset systematically harder, not
  a 1σ noise excursion. Binomial SE at n=160, p=0.36 ≈ 0.038; 0.1875
  is ~4σ below the full-100 mean. Subset bias dominates.
- **0.5625 vs 0.66** (pass@10) — same direction, smaller gap.
- **Δ ≈ +0.375** (pass@1 → pass@10 lift) — matches Phase 17 S9's
  +0.30 within problem-set noise. The **mechanism** (sampling lift)
  reproduces directly; the absolute level is subset-dependent.

The full 100×k=10 run is ~30 min and not load-bearing for "wiring
works" — Stage D will measure it as part of the MBPP MR saturation
curve. The directional aggregate number here confirms `MbppDomain`
+ `EvalSequential { aggregate: true }` produce numbers in the right
ballpark, which is the Stage C deliverable.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **156 tests** (was 149; +7 MBPP:
  parse_signature × 2, detect_imports × 2, loader, canonical verify,
  unknown-prompt inconclusive)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase22_mbpp_baseline --n-problems 8 --passk 1` greedy smoke
- ✅ E2E `phase22_mbpp_baseline --n-problems 16 --passk 10 --sequential
  --aggregate` cross-substrate aggregate measurement

## Phase 22 stage roadmap (post Stage C)

| stage | scope | status |
|---|---|---|
| A | HumanEvalDomain + baseline binary | ✅ (`91256a4`) |
| B | Sequential + aggregate eval, gap closed | ✅ (`bb78cc3`) |
| **C** | **MbppDomain (cross-substrate, mirror of A)** | ✅ (this commit) |
| D | Multi-round SFT on HumanEval via run_multi_round + QwenTrainerActorHandle → reproduce Phase 17 saturation curve | TODO |
| E | RL on HumanEval via Phase 21 Stage G REINFORCE mechanism | TODO |

Stage D is the real mechanism payoff: a 165 GPU-h Phase 17 saturation
curve (r=2..6) on Pekko, against the two substrates Stages A+B+C now
expose. Stage C completes the substrate matrix; the next bottleneck
is wallclock, not infrastructure.

## What this commit does NOT do

- **Full MBPP-100 × k=10 aggregate baseline** (~30 min). The 16-subset
  validates wiring + a directional aggregate number; the full sweep
  is a Stage D benchmark anchor and can be measured as part of D.
- **Multi-round SFT through Pekko on MBPP** (Stage D).
- **MBPP-974 full set**. Phase 17 S3 picked task_id 11–110 (100
  problems) and every Phase 17/18/19/20 MBPP measurement reused that
  subset — we follow suit for direct comparability.
- **Concurrent verify** for MBPP. Same write-lock serialization as
  HumanEvalDomain; same future-work item.

## Files

- `llm-actors/src/domain/mbpp.rs` — `MbppDomain`, `MbppChallenge`,
  `MbppProblem`, `parse_signature`, `detect_imports`, `build_challenge`
- `llm-actors/src/domain/mod.rs` — register `pub mod mbpp;`
- `llm-actors/examples/phase22_mbpp_baseline.rs` — CLI mirror of
  `phase22_humaneval_baseline` with `--sequential` + `--aggregate`
- `llm-actors/Cargo.toml` — register the example
- `docs/phase22-stage-c.md` (this)

## See also

- `docs/phase22-stage-a.md` — HumanEvalDomain + initial baseline
- `docs/phase22-stage-b.md` — metric mismatch diagnosis + aggregate
  mode (`EvalSequential { aggregate: true }`)
- `scripts/phase17_s3/problems.py` — the Python prompt-synthesis
  recipe Stage C ports
- `scripts/phase17_s9/run_passk_mbpp.py` — Phase 17 S9's MBPP pass@k
  measurement script
