#!/bin/bash
# Phase 18 S2 — rounds=3 SFT at HumanEval, seeds 0/1/2.
# Tests whether S1's multi-round compounding continues past round 2.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
for s in 0 1 2; do
  echo "=== p18s2 r3 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 3 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase18_s2/run_r3_seed${s}.json 2>&1 | tail -12
  echo "=== p18s2 r3 seed=$s done ==="
done
echo "=== r3_a done ==="
