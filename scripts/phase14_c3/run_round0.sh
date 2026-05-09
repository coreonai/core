#!/bin/bash
# Phase 14 C3: round-0-only DPO (β=0.1), SFT for r=1+, 5 seeds on GPU 3
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  echo "=== p14c3 round0 seed=$s gpu=3 ==="
  CUDA_VISIBLE_DEVICES=3 /tmp/p14_env/bin/python scripts/phase14_c3/self_improve.py \
    --seed $s --dpo-mode round0 --alpha 0.3 --beta 0.1 \
    --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -10
  echo "=== p14c3 round0 seed=$s done ==="
done
echo "=== round0 done ==="
