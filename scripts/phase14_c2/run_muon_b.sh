#!/bin/bash
# Phase 14 C2: Muon for LoRA, seeds 3, 4 on GPU 1
set -e
cd /raid/users/paul/workLLM
for s in 3 4; do
  echo "=== p14c2 muon seed=$s gpu=3 ==="
  CUDA_VISIBLE_DEVICES=3 /tmp/p14_env/bin/python scripts/phase14_c2/self_improve.py \
    --seed $s --optimizer muon --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -10
  echo "=== p14c2 muon seed=$s done ==="
done
echo "=== muon_b done ==="
