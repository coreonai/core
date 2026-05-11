#!/bin/bash
# Phase 17 S2 — restart label smoothing seeds 1, 2 (skipped seed 0 done).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
for s in 1 2; do
  echo "=== p17s2 ls0.1 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase17_s2/train_label_smooth.py \
    --seed $s --rounds 1 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --label-smoothing 0.1 2>&1 | tail -10
  echo "=== p17s2 ls0.1 seed=$s done ==="
done
echo "=== ls_a_redo done ==="
