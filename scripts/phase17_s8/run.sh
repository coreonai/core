#!/bin/bash
# Phase 17 S8 — pass@k extension at HumanEval, k=20.
# S6 found pass@10 = 0.524. Does scaling to k=20 push higher
# (saturating ceiling) or plateau (close to saturation already)?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
echo "=== p17s8 passk20 he gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_s6/run_passk.py \
  --seed 0 --k 20 --max-new-tokens 200 \
  --out scripts/phase17_s8/run_passk20_he.json 2>&1 | tail -25
echo "=== passk20_he done ==="
