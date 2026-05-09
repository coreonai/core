#!/bin/bash
# Phase 15 S4: re-test Muon at HumanEval substrate, seeds 3/4 on GPU passed as $1.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-3}
for s in 3 4; do
  echo "=== p15s4 muon seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s4/run_muon.py \
    --seed $s --optimizer muon --rounds 1 --samples 3 \
    --train-steps 200 --max-new-tokens 200 2>&1 | tail -8
  echo "=== p15s4 muon seed=$s done ==="
done
echo "=== muon_b done ==="
