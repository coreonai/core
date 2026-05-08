#!/bin/bash
# Phase 13 S1 A2 — AdamW × 5 seeds on GPU 0, sequential
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s1_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s1_adam_s${s}"
  echo "=== AdamW seed=$s ==="
  CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --optimizer adam 2>&1 | tail -8
  echo "=== adam seed=$s done ==="
  echo
done
echo "=== adam_seeds done ==="
