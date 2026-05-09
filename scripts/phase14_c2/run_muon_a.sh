#!/bin/bash
# Phase 14 C2: Muon for LoRA, seeds 0, 1, 2 on GPU 0
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2; do
  echo "=== p14c2 muon seed=$s gpu=2 ==="
  CUDA_VISIBLE_DEVICES=2 /tmp/p14_env/bin/python scripts/phase14_c2/self_improve.py \
    --seed $s --optimizer muon --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -10
  echo "=== p14c2 muon seed=$s done ==="
done
echo "=== muon_a done ==="
