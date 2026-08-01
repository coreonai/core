---
title: "Phase 22 §6.5 — BigCodeBench + Docker sandbox (investigation notes)"
date: "2026-08-01"
status: "investigation — not yet wired; adapter + prompt-source + sanity band are the next code steps"
---

# Why BigCodeBench, and where it sits

The workLLM benchmarking roadmap has two distinct bottlenecks, and they need
different instruments:

- **Contamination** → **LiveCodeBench** (temporal cutoff filtering; see
  `docs/phase22-rl-variance.md` neighbours and the LCB adapter). Answers: is a
  self-improve gain real generalization or polished pretraining recall?
- **Ceiling / realism** → **BigCodeBench**. Answers: how far can the recipe go
  on *harder, library-using, instruction-following* tasks that HumanEval's
  ceiling hides?

They are **complementary, not redundant**: LCB adds a *temporal* axis
(before/after cutoff), BigCodeBench adds a *difficulty/realism* axis. This note
is the investigation for standing BigCodeBench up.

# What BigCodeBench measures

- Self-contained functions that call **real libraries** (numpy, pandas,
  requests, …) and follow **complex instructions** — much higher ceiling than
  HumanEval.
- **Splits**: `Complete` (docstring-based, **completion-style** — suits a base
  model) vs `Instruct` (natural-language instructions).
- **Subsets**: `Full` = **1140 tasks**, `Hard` = **148 tasks**.
- **Metric**: **calibrated pass@1** — adds back missing setup (imports, global
  constants) the model skipped on long prompts before executing. This is the
  same class of concern as our `truncate_completion` handling, done on the
  harness side.

# The measurement discipline carries over

Same rules as the LCB adapter and the §6.5 guardrails:

- **Generate in Rust, score with the official harness.** Do **not** port a
  BigCodeBench scorer into Rust — that re-creates the exact silent-failure
  surface the `truncate_completion` bug came from
  (`docs/phase22-c4-c5-rl-vs-sft.md`, Lesson #6). Delegating to the upstream
  Docker harness keeps one ruler and makes numbers leaderboard-comparable.
- **Sanity anchor before trusting a number.** BigCodeBench has **no clean
  public BASE-model score** (the board is instruct-centric — see below), so it
  is a `PlausibilityBand` in `eval_sanity`, not a cited `Point`. The band only
  catches a broken harness (≈0) or an implausible score; it is not a
  leaderboard match.

# Custom-generation format (differs from LCB)

BigCodeBench ingests **JSONL** (one sample per line), not LCB's
`[{question_id, code_list}]` array:

```json
{"task_id": "BigCodeBench/N", "solution": "def task_func(...):\n    ...", "raw_solution": "<optional raw LLM output>"}
```

- `task_id` comes from `get_bigcodebench()` order → our `Domain::task_id` emits
  `"BigCodeBench/N"` (reuses the `task_id` infra already added for LCB).
- `solution` = the (truncated) completion; `raw_solution` optional.
- **Adapter implication**: `bench_export` needs a **second emitter**
  (`write_bigcodebench_jsonl`) alongside `write_lcb` — different shape (JSONL
  lines vs JSON array; `task_id`/`solution` vs `question_id`/`code_list`).

# Docker sandbox — required, and available here

**Why Docker**: generated code executes arbitrary libraries (numpy, pandas,
requests, filesystem, network) — isolation is mandatory. Scoring is
**CPU-bound** (running the Python tests), so it **does not compete with GPU
training** (only generation needs the GPUs).

**This cluster (checked 2026-08-01)**: Docker **26.1.3** daemon accessible, the
user is in the `docker` group (**no sudo needed**), storage driver `overlay2`,
**4.4 T free on `/raid`**. The official sandbox runs as-is.

**Local flow** (offline, reproducible):

```bash
# 1. sanitize / calibrate the generations
bigcodebench.syncheck --samples samples.jsonl        # -> samples-sanitized-calibrated.jsonl

# 2. score inside the official sandbox image
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest \
  --execution local --split complete --subset hard \
  --samples samples-sanitized-calibrated.jsonl
# results: *_eval_results.json, *_pass_at_k.json
```

**Alternatives**: `--execution gradio` (remote HF space, ~6–7 min Full /
~4–5 min Hard, no key) or `--execution e2b` (needs `E2B_API_KEY`). Local Docker
is best for reproducibility and offline runs.

# Public numbers (for the sanity band)

Leaderboard is **instruct-centric** — no clean base+completion point:

| model | split/subset | calibrated pass@1 |
|---|---|---|
| Qwen2.5-Coder-7B-**Instruct** | Complete / Full | **41.0%** |
| Qwen2.5-Coder-7B-**Instruct** | Complete / Hard | **18.2%** |
| Qwen2.5-Coder-7B-**Base** (ours) | Complete | *not published* |

Because `Complete` is docstring/completion-style, a **base** model is likely
*less* depressed than on LCB's chat format — but still below the instruct
neighbourhood. So the `eval_sanity` band uses the instruct 41.0% / 18.2% as the
upper neighbourhood and expects base below it; tighten after the first
calibrated run.

# Recommended sequencing

1. **Complete / Hard (148 tasks) first** — a cheap ceiling probe (far lighter
   than Full 1140), and the docstring split suits a base model.
2. **Infra**: `docker pull bigcodebench/bigcodebench-evaluate:latest` →
   smoke the eval on a tiny hand-made samples file → confirm the results JSON
   shape.
3. **Adapter**: `bench_export::write_bigcodebench_jsonl` + a BigCodeBench
   prompt-source in Rust (loads `task_id` + docstring prompt for generation).
4. **Sanity**: add BigCodeBench `PlausibilityBand` rows to `eval_sanity`
   (per subset), tighten after the first calibrated run.
5. If promising on Hard, extend to **Full 1140**.

# GPU-free next code steps (can run while a GPU wave holds the cards)

- `bench_export` BigCodeBench JSONL emitter (+ unit tests, exact-schema-keys
  assertion like the LCB one).
- `eval_sanity` BigCodeBench band rows.
- `docker pull` + a Docker eval smoke on a 2–3 line samples file.

The generation run itself (Rust → JSONL) needs the GPUs; scoring (Docker) does
not.

# Sources

- BigCodeBench repo — <https://github.com/bigcode-project/bigcodebench>
- ADVANCED_USAGE.md (custom `--samples` flow) —
  <https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md>
- bigcodebench PyPI — <https://pypi.org/project/bigcodebench/>
- BigCodeBench leaderboard — <https://bigcode-bench.github.io/>
- HF BigCodeBench blog — <https://github.com/huggingface/blog/blob/main/leaderboard-bigcodebench.md>
