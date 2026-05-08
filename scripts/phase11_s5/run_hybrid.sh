#!/bin/bash
# Phase 11 S5: hybrid SFT+DPO sweep on GPU 0 (sequential).
set -e
cd /raid/users/paul/workLLM

run_hybrid() {
  local alpha=$1
  local tag=$2
  echo "=== K9 hybrid SFT+DPO α=$alpha (β=0.1, frozen ref, 4 rounds) ==="
  CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "checkpoints/p11s5_$tag" \
    --seed-ckpt checkpoints/rust_seed.safetensors \
    --dpo-beta 0.1 \
    --dpo-reference-from checkpoints/rust_seed.safetensors \
    --dpo-sft-anchor-weight "$alpha" \
    --dpo-max-pairs-per-prompt 4 2>&1 | tail -15
  echo "=== p11s5_$tag done ==="
  echo
}

run_hybrid 0.3 hybrid_a03
run_hybrid 0.5 hybrid_a05
run_hybrid 0.7 hybrid_a07
echo "=== hybrid sweep done ==="
