#!/bin/bash
# Phase 19 S2 — Best-of-N (samples=10) + rounds=2 SFT at HumanEval.
# Tests if BoN harvest compounds with multi-round.
# Phase 17 S7a tested k=10 at rounds=1 (NEUTRAL Δ=+0.006).
# Phase 17 S1 found rounds=2 strong WIN at k=6 (+0.174). Combined?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 0 1 2; do
  echo "=== p19s2 bon_mr seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 2 --samples 10 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase19_s2/run_bon_mr_seed${s}.json 2>&1 | tail -12
  echo "=== p19s2 bon_mr seed=$s done ==="
done
echo "=== bon_mr_a done ==="
