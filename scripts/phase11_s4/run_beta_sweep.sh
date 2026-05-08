#!/bin/bash
# Phase 11 S4 — β sweep (frozen seed reference). Sequential on GPU 1.
set -e
cd /raid/users/paul/workLLM

run_dpo() {
  local beta=$1
  local tag=$2
  echo "=== K9 DPO β=$beta frozen ref (4 rounds) ==="
  CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "checkpoints/p11s4_$tag" \
    --seed-ckpt checkpoints/rust_seed.safetensors \
    --dpo-beta "$beta" \
    --dpo-reference-from checkpoints/rust_seed.safetensors \
    --dpo-max-pairs-per-prompt 4 2>&1 | tail -15
  echo "=== p11s4_$tag done ==="
  echo
}

run_dpo 0.01 b001
run_dpo 0.03 b003
run_dpo 0.05 b005
echo "=== beta sweep done ==="
