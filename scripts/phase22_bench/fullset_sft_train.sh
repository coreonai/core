#!/bin/bash
# Phase 22 §6.5 — re-train the FULL-SET HumanEval SFT recipe (the "hep" run,
# +0.106 on full HumanEval) for the LiveCodeBench generalization test. The
# original checkpoints were deleted; config RECOVERED from hep_seed42.log:
#   rounds=3, FULL 164 HumanEval (no --prompt-skip-list), samples-per-prompt=6,
#   train-steps=100, max-new=200, temp=0.8, top-k=0, lora r16/a32, lr=2e-4,
#   wd=0, batch=1, eval-n=64, eval-passk=5, 7B trainer-split.
# Training uses BF16 (HumanEval prompts are short — no long-prompt bug); the
# LCB eval of the resulting checkpoint uses F32 (long prompts).
#
# ~3.5 h/run (gen-dominated, 984 samples/round x 3), 2 GPUs/run.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_he_mr_sft
SEEDS=${1:-42,200}
if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build — rebuild with --features cuda." ; exit 1
fi
IFS=, read -r -a SEED_ARR <<< "$SEEDS"
gpu=0
for seed in "${SEED_ARR[@]}"; do
  gpu_m=$gpu; gpu_t=$((gpu + 1)); gpu=$((gpu + 2))
  CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
    --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
    --rounds 3 --samples-per-prompt 6 --train-steps 100 \
    --max-new-tokens 200 --temperature 0.8 --top-k 0 \
    --lora-rank 16 --lora-alpha 32 --lr 0.0002 --weight-decay 0 --batch-size 1 \
    --eval-n 64 --eval-passk 5 --seed "$seed" \
    --out-dir "scratch-7b-sft/hep_out_s${seed}" \
    --scratch-dir "scratch-7b-sft/hep_scr_s${seed}" \
    > "scratch-7b-sft/hep_seed${seed}.log" 2>&1 &
  echo "full-set SFT seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
done
echo "Launched full-set SFT x ${#SEED_ARR[@]} seeds. Checkpoints: hep_out_s{seed}/r0_merged.r2.safetensors"
