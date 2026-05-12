#!/bin/bash
# Phase 19 S1 — rounds=5 SFT at HumanEval, seeds 0/1/2.
# Tests S6's rounds=4 single-seed (+0.044) saturation question:
# does compounding plateau at r=5 or continue?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-0}
for s in 0 1 2; do
  echo "=== p19s1 r5 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 5 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase19_s1/run_r5_seed${s}.json 2>&1 | tail -14
  echo "=== p19s1 r5 seed=$s done ==="
done
echo "=== r5_a done ==="
