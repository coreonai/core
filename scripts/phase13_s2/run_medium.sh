#!/bin/bash
# Phase 13 S2 — medium (~10M, 4× scale) AdamW × 5 seeds on GPU 1.
# n_layer 4→8, n_head 4→8, n_embd 128→256, n_kv_head 4→4 (1× GQA).
# block_size stays 80 — corpus stays the same.
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s2_medium_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s2_medium_s${s}"
  echo "=== medium seed=$s ==="
  CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --n-layer 8 --n-head 8 --n-embd 256 --n-kv-head 4 \
    --optimizer adam 2>&1 | tail -8
  echo "=== medium seed=$s done ==="
done
echo "=== medium_seeds done ==="
