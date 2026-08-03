---
title: "Phase 22 §6.5 — external benchmarking guide (LiveCodeBench + BigCodeBench)"
date: "2026-08-03"
status: "reference — how to run the external code benchmarks, and what they found"
---

# Principle: generate in Rust, score with the official harness

One measurement ruler, no home-grown scorer. The Rust/Pekko stack **generates**
completions; the **official** harness scores them. Porting a scorer into Rust
re-creates the exact silent-failure surface that caused the C4/C5 corrections
(`FilteredDomain` truncation) — delegating to the upstream harness keeps numbers
directly leaderboard-comparable. This guide is the operational how-to; the
verdicts live in the per-benchmark notes.

# Shared components

| piece | file | role |
|---|---|---|
| dumper | `llm-actors/examples/phase22_dump_completions.rs` | `--benchmark {humaneval,mbpp,bigcodebench,livecodebench}` → generate → `--format {lcb,bigcodebench}` |
| export format | `llm-actors/src/bench_export.rs` | LCB `[{question_id, code_list}]`, BigCodeBench JSONL `{task_id, solution}` |
| generation-only domains | `llm-actors/src/domain/{livecodebench,bigcodebench}.rs` | load problems, build prompts, `task_id`; `verify` is an external-scoring stub |
| sanity anchor | `llm-actors/src/eval_sanity.rs` | cited-Point (HumanEval/MBPP) vs PlausibilityBand (LCB/BCB) |
| `Domain::task_id` | `llm-actors/src/domain/mod.rs` | benchmark-standard id keying the export |

**Two hard rules (see CLAUDE.md gotchas #10, #11):**
- **F32 for long prompts.** BF16 corrupts generation past ~500 tokens. LCB/BCB
  prompts are long → `--dtype f32` (the dumper default).
- **Aggregate pass@1, not greedy.** Self-improve gains live at aggregate pass@1
  (temp 0.8), not the single greedy mode. Use `--passk 5` and read the codegen
  aggregate pass@1, or a recipe's transfer will look like a loss.

# LiveCodeBench (contamination via contest-date split)

Scorer env (once): `scratch-7b-sft/tools/lcb-venv` (anaconda py3.12 + `lcb_runner`
`--no-deps` + `datasets==2.20` + numpy — the CLI wrapper's torch/vllm are not
needed for `codegen_metrics`).

```bash
# 1. export problems (release_v5 = 880, 2023-05..2025-01)
scratch-7b-sft/tools/lcb-venv/bin/python scripts/phase22_bench/lcb_export_problems.py release_v5
# 2. generate (F32; --checkpoint for a recipe) + 3. score + split, all in:
bash scripts/phase22_bench/lcb_run_base.sh                       # base, greedy
bash scripts/phase22_bench/lcb_run_recipe.sh <ckpt> <out> <gpu>  # recipe, greedy
bash scripts/phase22_bench/lcb_agg_verify.sh                     # base+recipe, AGGREGATE pass@1  ← the real test
# score directly:
scratch-7b-sft/tools/lcb-venv/bin/python scripts/phase22_bench/lcb_score.py \
  --gens <path.json> --release release_v5 [--start-date 2024-09-01 | --end-date 2024-09-01]
```

Cutoff `2024-09-01` (Qwen2.5-Coder-7B ~release). Detail: `phase22-livecodebench-notes.md`.

# BigCodeBench (ceiling / real-library tasks) — base measured

JSONL `{task_id, solution}` (not LCB's array); `solution` = completion body,
`--calibrated True` prepends the prompt. Scoring in the official Docker sandbox
(`bigcodebench/bigcodebench-evaluate:latest`, bcb 0.2.4 / dataset v0.1.4,
`--execution local`; cluster has Docker 26.1.3 + the docker group).

```bash
# 1. export Hard v0.1.4 (148) + 2. generate base (F32, 8-GPU) + 3. Docker score:
bash scripts/phase22_bench/bcb_run_base.sh      # -> bcb_hard_base_samples.jsonl
bash scripts/phase22_bench/bcb_score.sh         # complete hard, --execution local
# score any samples file:
bash scripts/phase22_bench/bcb_score.sh <samples.jsonl>
```

Docker gotcha: image runs as `bigcodebenchuser` → pass
`--user $(id -u):$(id -g) -e HOME=/app/.dockerhome` (writable results + runtime
dataset cache). Complete/Hard first. Detail: `phase22-bigcodebench-notes.md`,
`llm.coreon.build/bigcodebench.html`.

# Findings (index)

- **BF16 long-prompt bug found + fixed** (`39f038e`) — the biggest by-product.
- **LiveCodeBench base validated** (aggregate pass@1 overall 0.075, post-cutoff
  0.041); pipeline works, prompt format matches.
- **Self-improve GENERALIZES** (correct metric, 6-seed confirmed): both HumanEval
  recipes lift LCB post-cutoff (unseen) at aggregate pass@1. Full-set SFT
  0.0413 → 0.0562 ± 0.006 (**+0.0149, +2.5σ**); **K=8 RL 0.0413 → 0.1105 ± 0.012
  (+0.0692, +5.7σ), ~2× SFT** and 6/6 seeds beat both base and SFT. K=8 RL's
  lift is almost pure post-cutoff (pre near-flat +0.010) — the cleanest
  generalization signature, and it *inverts* the in-domain SFT-over-RL
  deployment verdict for the transfer objective. Greedy was the wrong ruler
  (looked like a loss). Detail: `phase22-livecodebench-notes.md`,
  `llm.coreon.build/livecodebench.html`.
- **BigCodeBench base measured** (Complete/Hard, calibrated pass@1): base 7B
  **0.169** (gt 0.973), just below the instruct neighbourhood 0.182 — pipeline
  validated end-to-end via the Docker sandbox; format smoke-confirmed
  (completion body + `--calibrated`). Recipe ceiling probe is the follow-up.
  Detail: `phase22-bigcodebench-notes.md`, `llm.coreon.build/bigcodebench.html`.
- **RL variance study** (separate, concluded): `phase22-rl-variance.md`,
  `llm.coreon.build/rl-variance.html`.

# Reproducing a recipe checkpoint

Full-set HumanEval SFT (the "hep" recipe, +0.10 at aggregate pass@1) config was
recovered from `hep_seed42.log` and shipped:
`scripts/phase22_bench/fullset_sft_train.sh` (rounds=3, full 164,
samples-per-prompt=6, train-steps=100, lr 2e-4, lora r16/α32). **Keep the launch
command** — the original hep checkpoints were deleted and their command did not
survive, costing a full re-train.
