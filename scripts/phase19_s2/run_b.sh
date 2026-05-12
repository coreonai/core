#!/bin/bash
# Phase 19 S2 — BoN k=10 + rounds=2, seeds 3/4.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-6}
for s in 3 4; do
  echo "=== p19s2 bon_mr seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 2 --samples 10 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase19_s2/run_bon_mr_seed${s}.json 2>&1 | tail -12
  echo "=== p19s2 bon_mr seed=$s done ==="
done
echo "=== bon_mr_b done ==="
