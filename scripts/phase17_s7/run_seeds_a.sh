#!/bin/bash
# Phase 17 S7a — samples=10 SFT at HumanEval, seeds 0/1/2.
# Tests training-axis: does k=10 chosen-pool expansion lift pass@1?
# Phase 16 S1 samples=6 baseline: 0.230 ± 0.031.
# 2σ threshold 0.062 → win at samples=10 final > 0.292.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 0 1 2; do
  echo "=== p17s7a s10 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 10 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase17_s7/run_s10_seed${s}.json 2>&1 | tail -10
  echo "=== p17s7a s10 seed=$s done ==="
done
echo "=== s10_a done ==="
