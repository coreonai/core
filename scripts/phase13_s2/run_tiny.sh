#!/bin/bash
# Phase 13 S2 — tiny (~1M, default) AdamW × 5 seeds on GPU 0
# Same as Phase 13 S1 AdamW BUT now binary uses 10 challenges
# (Phase 13 S1 used 3-challenge binary).
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s2_tiny_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s2_tiny_s${s}"
  echo "=== tiny seed=$s ==="
  CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --optimizer adam 2>&1 | tail -8
  echo "=== tiny seed=$s done ==="
done
echo "=== tiny_seeds done ==="
