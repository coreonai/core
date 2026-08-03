#!/bin/bash
# Phase 22 §6.5 — K=8 RL (hard-tail REINFORCE) aggregate LCB transfer.
# Follow-up to the full-set SFT 6-seed generalization result: measure the K=8 RL
# recipe at the CORRECT metric (aggregate pass@1, passk 5, temp 0.8, F32) — the
# earlier K=8 LCB numbers were greedy (superseded). Same slices as the SFT
# aggregate run (offsets 640/670/700/730 x 30 = 120 problems), same F32 path.
# Checkpoints: scratch-7b-sft/rlvar_k8/mean_k8_seed{S}_final.safetensors (6 seeds).
# Score pre/post cutoff per seed; summarise 6-seed post-cutoff vs base 0.0413
# and vs full-set SFT 0.0562.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
SEEDS=(42 100 200 300 400 500)

gen_seed() { # $1=seed $2=gpubase -> 4 slices, passk 5, F32, K=8 RL checkpoint
  local s=$1 g=$2
  local OUT="scratch-7b-sft/lcb_k8agg_s${s}"; mkdir -p "$OUT"
  local CKPT="scratch-7b-sft/rlvar_k8/mean_k8_seed${s}_final.safetensors"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; return 1; }
  for off in 640 670 700 730; do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 --checkpoint "$CKPT" --offset $off --n-problems 30 --passk 5 \
      --max-new-tokens 768 --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
}

# generate for the 6 seeds, 2 at a time (8 GPUs).
n=${#SEEDS[@]}; idx=0
while [ $idx -lt $n ]; do
  a=${SEEDS[$idx]}; b=${SEEDS[$((idx+1))]}
  echo "=== K8 LCB gen batch: $a (GPU0-3)${b:+ + $b (GPU4-7)} ==="
  gen_seed "$a" 0
  [ -n "$b" ] && gen_seed "$b" 4
  wait
  idx=$((idx + 2))
done
echo "=== all K8 LCB generation done; scoring ==="

score() { # $1=gensdir  -> overall/pre/post aggregate pass@1
  local OUT=$1
  python3 -c "
import json, glob
g=[]
for f in sorted(glob.glob('$OUT/slice_*.json')): g+=json.load(open(f))
json.dump(g, open('$OUT/gens_all.json','w'))
" 2>/dev/null
  for w in "overall:" "pre:--end-date $CUT" "post:--start-date $CUT"; do
    nm=${w%%:*}; flag=${w#*:}
    val=$($V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 $flag 2>/dev/null | grep -aoE "pass@1 \(codegen_metrics\): [0-9.]+" | grep -oE "[0-9.]+$")
    echo "  $nm = $val"
  done
}
for s in "${SEEDS[@]}"; do
  echo "############ K8 RL seed $s ############"; score "scratch-7b-sft/lcb_k8agg_s${s}"
done
echo "=== LCB_K8_AGG_COMPLETE ==="
