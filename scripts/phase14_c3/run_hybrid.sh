#!/bin/bash
# Phase 14 C3: hybrid SFT+DPO (α=0.3, β=0.1), 5 seeds on GPU 2
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  echo "=== p14c3 hybrid seed=$s gpu=2 ==="
  CUDA_VISIBLE_DEVICES=2 /tmp/p14_env/bin/python scripts/phase14_c3/self_improve.py \
    --seed $s --dpo-mode hybrid --alpha 0.3 --beta 0.1 \
    --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -10
  echo "=== p14c3 hybrid seed=$s done ==="
done
echo "=== hybrid done ==="
