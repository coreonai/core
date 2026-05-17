#!/bin/bash
# Phase 22 Stage D G5 — full Phase 17 recipe match
#   = mask-prompt fix (`bc90db5`) + cosine LR schedule (`c7a7aed`)
#
# 5 seeds × r=2 × gen-n=164 × eval-n=32 × passk=3 × train-steps=30 on
# HumanEval. Launches in parallel on GPUs 0/1/5/6/7.
#
# Run AFTER `cargo build -p llm-actors --example phase22_he_mr_sft
# --features cuda --release` (with the c7a7aed binary).
#
# Wallclock: ~30 min total.
set -e
cd /raid/users/paul/workLLM

# Sanity check the binary has the recent changes.
if ! strings ./target/release/examples/phase22_he_mr_sft | grep -q "cosine_warmup_lr\|train_sft_pairs"; then
  echo "⚠ binary is missing the c7a7aed (cosine) or bc90db5 (mask) symbols"
  echo "  Rebuild: CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:\$PATH \\"
  echo "             cargo build -p llm-actors --example phase22_he_mr_sft --features cuda --release"
  exit 1
fi

for i in 0 1 2 3 4; do
  case $i in
    0) gpu=5 ;; 1) gpu=6 ;; 2) gpu=7 ;; 3) gpu=1 ;; *) gpu=0 ;;
  esac
  seedval=$((i*100+100))
  mkdir -p /tmp/phase22d_G5_seed${i}
  CUDA_VISIBLE_DEVICES=$gpu ./target/release/examples/phase22_he_mr_sft \
    --seed $seedval --rounds 2 --gen-n 164 --eval-n 32 --eval-passk 3 \
    --train-steps 30 --max-new-tokens 200 \
    --scratch-dir /tmp/phase22d_G5_seed${i}/scratch \
    --out-dir /tmp/phase22d_G5_seed${i}/ckpts \
    > /tmp/phase22d_G5_seed${i}/run.log 2>&1 &
  echo "G5 seed=$seedval GPU $gpu PID=$!"
done
echo "G5 launched. Wait with: until [ \"\$(pgrep -af 'phase22_he_mr_sft --seed' | wc -l)\" -eq 0 ]; do sleep 60; done"
