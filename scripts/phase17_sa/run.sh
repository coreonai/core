#!/bin/bash
# Phase 17 SA — Multi-round + pass@k eval (single seed proof).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p17sa mr_passk seed=0 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_sa/run_mr_passk.py \
  --seed 0 --rounds 2 --samples 6 --passk-k 10 --train-steps 200 --max-new-tokens 200 \
  2>&1 | tail -25
echo "=== sa done ==="
