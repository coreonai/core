---
title: "Phase 22 — Qwen2.5-Coder-7B deployment recipe"
subtitle: "Cost × quality selector, distilled from the measured 7B study"
date: "2026-08-25"
status: "DRAFT — one cell pending the C-1 confirmatory result"
---

# What this is

The 7B counterpart to `docs/phase20-deployment-recipe.md` (which covers
0.5B). **No new measurement** — this is the distillation step. Every number
below is taken from a measured run on one consistent ruler; sources are cited
per table.

Read this if you need to pick a recipe. Read
`docs/phase22-c4-c5-rl-vs-sft.md` and
`docs/phase22-livecodebench-notes.md` if you need to know why.

# TL;DR — recipe selector

| budget | recipe | in-domain pass@1 | transfer (LCB post) | train GPU-h |
|---|---|---:|---:|---:|
| **free** | base + pass@k | 0.172 → 0.422 @ k=5 | 0.041 | **0** |
| **cheap** | SFT — hard tail, s=16 | 0.364 | *not measured* | ~11 |
| **cheap** | SFT — full set | *not measured* | 0.056 | ~11 |
| **balanced** | **RL K=8** | 0.538 | 0.111 | ~36 |
| **transfer-max** | **RL K=16** | 0.572 | **0.129** | ~70 |
| **in-domain-max** | **RL K=32** | **0.606** | 0.128 | ~78 |

⚠ **Two different SFT arms appear below and they are not interchangeable.**
The in-domain numbers come from *hard-tail SFT* (samples=16, idx 100–163);
the transfer numbers come from *full-set SFT* (all 164). Each was only ever
measured on its own axis, so there is no row where one SFT recipe has both.
Do not read across.

**The single most important line**: if you only ever do one thing, do
**pass@k**. It is free, needs no training, and on the full HumanEval set it
takes aggregate pass@1 from 0.366 to a per-prompt pass@10 of 0.713
(**+0.347**). It is orthogonal to and additive with everything below.

# Substrate

- **Model**: Qwen2.5-Coder-7B (HF), BF16 training / F32 long-prompt inference
- **Adapter**: LoRA r=16, α=32, `q_proj` + `v_proj`
- **Training**: AdamW lr=2e-4, cosine schedule with 10% warmup,
  **completion-only loss** (`sft_mask_prompt=true`)
- **RL**: `--advantage-mode mean --pg-positive-only --rl-steps 30
  --sync-every 1 --pg-micro-batch-size 1`
- **Hardware**: A100-40GB. RL needs **2 GPUs per run** (model ~15 GB +
  policy-gradient trainer ~20-27 GB); use `--trainer-gpu`.

⚠ **F32 for long prompts.** BF16 silently corrupts generation past ~500
prompt tokens. HumanEval/MBPP are short enough to hide it; LiveCodeBench and
BigCodeBench are not. See CLAUDE.md gotcha #10.

# Quality, measured

## In-domain (HumanEval hard tail, idx 100–163, 64 problems)

6 seeds per arm except SFT (4). Same ruler throughout.

| recipe | pass@1 | σ | pass@5 |
|---|---:|---:|---:|
| base | 0.172 | — | 0.422 |
| SFT — hard tail, s=16, r=2 | 0.364 | 0.037 | 0.566 |
| RL K=2 | 0.298 | 0.098 | 0.531 |
| RL K=4 | 0.399 | 0.090 | 0.568 |
| RL K=8 | 0.538 | 0.076 | 0.656 |
| RL K=16 | 0.572 | 0.069 | 0.669 |
| **RL K=32** | **0.606** | **0.037** | 0.661 |

Source: `phase22-c4-c5-rl-vs-sft.md`, C-2 in-domain sweep.

## Transfer (LiveCodeBench post-cutoff, unseen, n=92)

The honest generalization number — problems released after the model's
training cutoff.

| recipe | aggregate pass@1 | Δ base |
|---|---:|---:|
| base | 0.041 | — |
| full-set SFT | 0.056 | +0.015 |
| RL K=2 | 0.084 | +0.043 |
| RL K=4 | 0.098 | +0.057 |
| RL K=8 | 0.111 | +0.069 |
| **RL K=16** | **0.129** | **+0.088** |
| RL K=32 | 0.128 | +0.087 |

Source: `phase22-livecodebench-notes.md`.

## Difficulty ceiling (BigCodeBench Complete/Hard, 148 tasks)

| recipe | aggregate pass@1 | σ |
|---|---:|---:|
| base | 0.146 | — |
| SFT | 0.155 | 0.021 |
| RL K=8 | 0.181 | 0.016 |
| RL K=16 | 0.179 | **0.010** |

Source: `bigcodebench.html`, C-3.

# Cost, measured

## Training

Wall clock per RL step on 64 prompts, measured across the sweep. Generation
does **not** scale linearly with K — fixed per-step cost dominates at large K,
which is why K=32 costs only ~1.1× K=16 rather than 2×.

| recipe | per step | 30 steps wallclock | GPU-h (2 GPU/run) |
|---|---:|---:|---:|
| SFT — hard tail, s=16, r=2 | — | ~5.5 h | ~11 |
| RL K=2 | 8 min | 4.0 h | 8 |
| RL K=4 | 15 min | 7.5 h | 15 |
| RL K=8 | 36 min | 18.0 h | 36 |
| RL K=16 | 70 min | 35.0 h | 70 |
| RL K=32 | 78 min | 39.0 h | 78 |

Multiply by the number of seeds if you train several. Four runs fit on 8 GPUs.

## Inference (pass@k)

pass@k needs k samples per prompt; cost is linear in k and orthogonal to
training. On a 7B at max_new=192, F32, a single sample is ~3–4 s warm.

# The Pareto front

Plotting quality against training GPU-hours, the non-dominated set is:

1. **base + pass@k** — 0 GPU-h. Nothing else is free.
2. **SFT** — ~11 GPU-h. In-domain 0.364 (hard-tail arm) / transfer 0.056
   (full-set arm). Cheapest training either way.
3. **RL K=8** — 36 GPU-h for 0.538 in-domain / 0.111 transfer. The knee.
4. **RL K=16** — 70 GPU-h, best transfer (0.129).
5. **RL K=32** — 78 GPU-h, best in-domain (0.606) and tightest σ (0.037).

**RL K=2 and K=4 are dominated on quality** (both below SFT + a fraction of
K=8) but remain the cheapest RL entry points if 36 GPU-h is out of reach.

**The knee is K=8.** It buys 84% of K=32's in-domain gain for 46% of the
compute. Past it you are paying ~2× for the last ~12%.

# How to choose

**Pick by which axis you actually care about — they diverge.**

| you want | pick | why |
|---|---|---|
| Anything at zero cost | base + pass@k | +0.347, training-free |
| Generalization to unseen problems | **RL K=16** | transfer saturates here; K=32 buys nothing (−0.001) |
| Best score on the distribution you trained on | **RL K=32** | in-domain keeps rising past the transfer ceiling |
| Lowest variance across seeds | **RL K=32** (σ 0.037) or SFT (σ 0.037) | RL K=32 matches SFT's σ at 1.7× its mean |
| Least compute that still beats SFT | **RL K=2** | beats *full-set* SFT in 6/6 seeds on transfer, at 8 GPU-h |
| Difficulty ceiling (library-heavy tasks) | RL K=8 or K=16 | harvest width does not help this axis |

# Traps this study fell into — do not repeat

1. **Match the metric to where the headroom is.** A saturated metric hides
   real gains. pass@5 read "flat" on 7B HumanEval while pass@1 showed +0.106.
2. **One ruler.** Two eval paths gave 0.246 and 0.422 for the *same* base.
   Never compare a filtered/subset number to an unfiltered one; re-measure
   both on the same path.
3. **The trend, not the pairs.** In the K sweep, three consecutive pairwise
   comparisons read "no effect" (t≈1.5) while the 24-run trend was decisive
   (t=3.68). Budget for the whole sweep.
4. **Benchmark-axis dependence is real.** Harvest width is worth +0.019 on
   LCB and 0.000 on BigCodeBench. A recipe gain measured on one axis does not
   transfer to another.
5. **Verify the binary is a CUDA build.** `cargo test --workspace` rebuilds
   examples as CPU. Check `strings <binary> | grep -c cudarc` (74 vs 0), not
   the timestamp.

# Open

- **`--pg-positive-only` justification** — *pending C-1*. The in-domain
  advantage (+0.124 pass@1) came from optional stopping and is under
  pre-registered replication (`phase22-c1-prereg.md`, n=12). It is null on
  transfer (t=−0.52), so if C-1 comes back null the flag stays default on the
  variance argument alone (full-advantage's spread is 2.4× wider). **This
  cell will be filled when C-1 reports; it does not change any number above.**
- **K=32's σ** rests on n=6 and no pairwise step is significant
  (F=4.34, p≈0.13). More seeds would firm it.
- **The two SFT arms have never been cross-measured.** Hard-tail SFT has no
  transfer number and full-set SFT has no in-domain hard-tail number, so the
  SFT baseline in this document is axis-dependent. Scoring the existing
  `htr_out_s*` checkpoints on LCB would close this with no training.
- **Multi-round RL** untested. All RL numbers are single-round, 30 steps.
