---
title: "Phase 22 §6.5 — BigCodeBench + Docker sandbox (investigation notes)"
date: "2026-08-01"
status: "base + recipe ceiling probe DONE — both recipes lift aggregate pass@1 (+0.02–0.03); K8's LCB dominance does NOT reproduce here (K8≈SFT within noise)"
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

# Base result (MEASURED) — Complete / Hard, calibrated pass@1

| model | split / subset | calibrated pass@1 | gt pass rate |
|---|---|---|---|
| **Qwen2.5-Coder-7B base** (ours) | Complete / Hard (148) | **0.169** | 0.973 |
| Qwen2.5-Coder-7B-Instruct (ref) | Complete / Hard | 0.182 | — |

- **Pipeline validated end-to-end.** Generate in Rust (F32, greedy, 148 tasks,
  8-GPU) → merge to `{task_id, solution}` JSONL → official Docker sandbox
  (`bigcodebench.evaluate complete hard --execution local --calibrated True`).
  Bundled tool **bcb 0.2.4**, dataset **v0.1.4** (matches our export).
- **Format confirmed by a known-correct smoke.** `solution` = the model's
  completion *body*; `calibrated=True` prepends the prompt. Canonical solutions
  fed as samples score **pass@1 1.000** — so our completion-body samples are the
  right shape (the `--calibrated` convention, not full-program).
- **Sane, not depressed.** 0.169 lands **just below the instruct neighbourhood
  0.182** — exactly the notes' prediction (Complete is docstring/completion-style,
  so a base model is *less* depressed than on LCB's chat format). Not ≈0 (harness
  works), below instruct as expected. `eval_sanity` Hard band recalibrated to the
  measured value: `[0.09, 0.25]` (was `[0.0, 0.22]`).
- **4 groundtruth tasks fail in-sandbox** (590, 418, 509, 417 — network/env
  deps like live FTP), so gt pass rate is 0.973, not 1.0; standard for Hard.
- **Reference to beat**: base Hard calibrated pass@1 = **0.169**. A self-improve
  recipe's Hard number is the ceiling probe (follow-up; base-first was the plan).

Commands: `scripts/phase22_bench/bcb_run_base.sh` (generate),
`bcb_score.sh` (Docker score), `bcb_smoke.py` (format smoke).

# Recipe ceiling probe (MEASURED) — aggregate pass@1, both recipes lift; K8 ≈ SFT here

Measured at **aggregate pass@1 (temp 0.8, passk 5, F32)** — the metric where
self-improve lives (lesson #11), not greedy. base + full-set SFT + K=8 RL, 3
seeds (42, 100, 200) each on Hard:

| arm | pass@1 (aggregate) | pass@5 | Δ pass@1 vs base |
|---|---|---|---|
| base | 0.1459 | **0.4324** | — |
| full-set SFT (3-seed) | 0.1676 ± 0.0082 | 0.3941 | **+0.0216** |
| K=8 RL (3-seed) | 0.1739 ± 0.0163 | 0.3829 | **+0.0279** |

Per-seed pass@1 — SFT: 0.1635 / 0.1622 / 0.1770; K8: 0.1797 / 0.1554 / 0.1865.

- **Both recipes lift the ceiling at aggregate pass@1.** base 0.146 → SFT 0.168 →
  K8 0.174 (+0.022 / +0.028). Self-improve pushes BigCodeBench Hard, the harder
  benchmark, not just HumanEval — a genuine (if modest) ceiling lift.
- **K=8 RL's LiveCodeBench dominance does NOT reproduce here.** K8 beats SFT by
  only **+0.006 (+0.34σ pooled)** — within noise — and K8 is *higher*-variance
  (σ 0.016 vs 0.008; seed 100 = 0.155 falls below every SFT seed). Contrast LCB,
  where K8 beat SFT by +4σ. **Recipe superiority is benchmark-axis-dependent**:
  on LCB's *contamination* axis K8 dominates; on BigCodeBench's *difficulty* axis
  K8 ≈ SFT.
- **Clearest sharpening signature in the project.** pass@5 *decreases* with
  self-improve: base 0.432 > SFT 0.394 > K8 0.383. The recipes concentrate
  probability mass onto the mode (pass@1 ↑, diversity ↓) — lesson #11's mechanism
  observed directly. (So base's *greedy* 0.169 > base *aggregate* 0.146: greedy
  picks the strong mode; the recipes lift the aggregate back to ≈ base-greedy.)
- **Caveats.** 3 seeds; small absolute deltas (a few problems on 148); gt pass
  rate 0.980. Directional, not a tight CI. One OOM recovery: 3 arms' slice_19
  hit CUDA OOM under checkpoint-load + F32 passk5 contention → regenerated alone
  on a dedicated GPU (`bcb_probe_recover.sh`).

Command: `scripts/phase22_bench/bcb_recipe_probe.sh` (generate all arms at
aggregate + Docker-score + summarize).

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

# Implementation status

GPU-free code — **done** (Phase 22 §6.5 commits):

- [x] `bench_export::write_bigcodebench_jsonl` + `bigcodebench_entries`
      (`BcbEntry {task_id, solution, raw_solution?}`), unit-tested.
- [x] `domain::bigcodebench::BigCodeBenchDomain` — generation-only prompt
      source (Complete/Instruct split; `verify` is an external-scoring stub;
      `task_id → "BigCodeBench/N"`).
- [x] `phase22_dump_completions` example — `--benchmark bigcodebench
      --format bigcodebench` generates + writes the JSONL.
- [x] `eval_sanity` BigCodeBench `PlausibilityBand` rows, keyed
      `BigCodeBench-Complete-{Full,Hard}` (instruct neighbourhood 41.0% / 18.2%
      as the upper guard).

Data / Docker / GPU — **done**:

- [x] Export HF `bigcode/bigcodebench-hard` v0.1.4 →
      `data/bigcodebench/BigCodeBench-Hard.jsonl` (148 tasks, Complete).
- [x] `docker pull bigcodebench/bigcodebench-evaluate:latest` (15.1 GB) + a
      known-correct smoke (canonical solutions → pass@1 1.000) confirming the
      completion-body + `--calibrated` format and the results JSON shape.
- [x] Base generation (Rust → JSONL, F32, 8-GPU) → Docker `--execution local`
      → **calibrated pass@1 0.169** → Hard band tightened to `[0.09, 0.25]`.

Docker gotcha: the eval image runs as `bigcodebenchuser`; pass
`--user $(id -u):$(id -g) -e HOME=/app/.dockerhome` so results files are writable
and the runtime dataset cache lands in the mount. The generation run needs the
GPUs; scoring (Docker) is CPU-bound and does not.

Follow-up (not yet run): score a self-improve recipe (full-set SFT and/or K=8 RL)
on Hard for the ceiling probe; extend to Full 1140 if promising.

# Sources

- BigCodeBench repo — <https://github.com/bigcode-project/bigcodebench>
- ADVANCED_USAGE.md (custom `--samples` flow) —
  <https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md>
- bigcodebench PyPI — <https://pypi.org/project/bigcodebench/>
- BigCodeBench leaderboard — <https://bigcode-bench.github.io/>
- HF BigCodeBench blog — <https://github.com/huggingface/blog/blob/main/leaderboard-bigcodebench.md>
