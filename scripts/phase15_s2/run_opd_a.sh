#!/bin/bash
# Phase 15 S2: OPD student, seeds 0/1/2 on GPU 2
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2; do
  echo "=== p15s2 opd seed=$s gpu=2 ==="
  CUDA_VISIBLE_DEVICES=2 /tmp/p14_env/bin/python \
    scripts/phase15_s2/self_improve_opd.py \
    --seed $s --rounds 1 --samples 3 --train-steps 200 \
    --opd-temperature 2.0 --kl-direction forward \
    --max-new-tokens 200 \
    --specialists-dir checkpoints/phase15_s2 \
    2>&1 | tail -15
  echo "=== p15s2 opd seed=$s done ==="
done
echo "=== opd_a done ==="
