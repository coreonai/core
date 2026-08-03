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

# BigCodeBench (ceiling / real-library tasks) — infra ready, not yet run

JSONL `{task_id, solution}` (not LCB's array). Scoring in the official Docker
sandbox (`bigcodebench/bigcodebench-evaluate:latest`, `--execution local`;
this cluster has Docker 26.1.3 + the docker group). Complete/Hard first.
Detail + commands: `phase22-bigcodebench-notes.md`,
`llm.coreon.build/bigcodebench.html`.

# Findings (index)

- **BF16 long-prompt bug found + fixed** (`39f038e`) — the biggest by-product.
- **LiveCodeBench base validated** (aggregate pass@1 overall 0.075, post-cutoff
  0.041); pipeline works, prompt format matches.
- **Self-improve GENERALIZES** (correct metric, 6-seed confirmed): full-set SFT
  lifts LCB post-cutoff (unseen) 0.0413 → 0.0562 ± 0.006 at aggregate pass@1
  (**Δ +0.0149, +2.5σ, 6/6 seeds beat base**) → real learning, not recall.
  Greedy was the wrong ruler (looked like a loss). Detail:
  `phase22-livecodebench-notes.md`, `llm.coreon.build/livecodebench.html`.
- **RL variance study** (separate, concluded): `phase22-rl-variance.md`,
  `llm.coreon.build/rl-variance.html`.

# Reproducing a recipe checkpoint

Full-set HumanEval SFT (the "hep" recipe, +0.10 at aggregate pass@1) config was
recovered from `hep_seed42.log` and shipped:
`scripts/phase22_bench/fullset_sft_train.sh` (rounds=3, full 164,
samples-per-prompt=6, train-steps=100, lr 2e-4, lora r16/α32). **Keep the launch
command** — the original hep checkpoints were deleted and their command did not
survive, costing a full re-train.
