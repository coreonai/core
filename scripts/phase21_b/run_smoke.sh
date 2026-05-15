#!/bin/bash
# Phase 21 Stage B — Substrate scale-up + pass@k lift measurement.
#
# At n_embd=512 / n_layer=6 (~24M params, ~25× K9 1M baseline) does
# inference-time pass@k surface lift the way Phase 17 S6 measured
# against Qwen (pass@10 0.524 vs pass@1 0.216)?
#
# Args:
#   $1 = GPU index (default 5)
#   $2 = passk (1 = greedy baseline, ≥2 = inference scaling)
#
# When passk ≥ 2 the eval automatically bumps to temp=0.8 / top-k=10
# so the k samples diverge (passk > 1 with greedy temp=0 produces k
# identical samples — useless for measuring pass@k lift).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
PASSK=${2:-5}

if [[ "$PASSK" -gt 1 ]]; then
  EVAL_TEMP=0.8
  EVAL_TOPK=10
else
  EVAL_TEMP=0.0
  EVAL_TOPK=1
fi

OUT=scripts/phase21_b/run_n512_l6_passk${PASSK}.log
echo "=== p21b scale-up passk=$PASSK eval_temp=$EVAL_TEMP gpu=$GPU ===" | tee $OUT
CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  ./target/release/examples/self_improve_rust \
  --rounds 2 \
  --pretrain-steps 3000 \
  --pretrain-examples 600 \
  --gen-n 24 \
  --eval-n 24 \
  --round-train-steps 600 \
  --eval-passk $PASSK \
  --eval-temperature $EVAL_TEMP \
  --eval-top-k $EVAL_TOPK \
  --n-embd 512 \
  --n-layer 6 \
  --n-head 8 \
  --n-kv-head 4 \
  --seed-ckpt checkpoints/rust_seed_b_passk${PASSK}.safetensors \
  --round-ckpt checkpoints/rust_round_b_passk${PASSK} \
  --scratch-dir /tmp/workllm-rust-scratch-b-passk${PASSK} \
  2>&1 | tee -a $OUT
echo "=== p21b done passk=$PASSK ===" | tee -a $OUT
