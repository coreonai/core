#!/bin/bash
# Phase 20 S1 — rounds=6 SFT at HumanEval, seed 0.
# Phase 19 S1 mean 0.556 ± 0.037 at r=5, Δ=+0.037 vs r=4.
# Question: does r=6 plateau? Decision gates in docs/phase20-design.md.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-0}
echo "=== p20s1 r6 seed=0 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase15_s1/self_improve.py \
  --seed 0 --rounds 6 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase20_s1/run_r6_seed0.json 2>&1 | tail -16
echo "=== p20s1 r6 seed=0 done ==="
