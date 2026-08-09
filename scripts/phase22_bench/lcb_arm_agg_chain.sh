#!/bin/bash
# Phase 22 §6.5 — aggregate LCB transfer for an ARBITRARY RL arm.
#
# Generalises lcb_k8_agg_chain.sh (which hardcodes the K=8 posonly arm) so a
# second arm can be scored on the identical ruler: same slices
# (offsets 640/670/700/730 x 30 = 120 problems), same passk 5 / temp 0.8 /
# F32 path, same pre/post cutoff split. F32 is mandatory — BF16 corrupts
# generation past ~500 prompt tokens (CLAUDE.md gotcha #10), and LCB prompts
# are 500-1000+.
#
# Motivation: every K=8 transfer run so far used --pg-positive-only, so
# "does bounding the objective matter for TRANSFER?" has never been measured.
# The C4 8-seed comparison settled it in-domain at pass@1 (+0.124, 8/8,
# p=0.0086) but transfer is a different axis.
#
# Usage: lcb_arm_agg_chain.sh <ckpt_dir> <tag> <seeds> [out_prefix]
#   lcb_arm_agg_chain.sh scratch-7b-sft/rlvar_k8_fulladv mean_k8_fulladv 42,100,200,300
#
# Reference (posonly K=8, 6 seeds, post-cutoff n=92 aggregate pass@1):
#   base 0.0413 | SFT 0.0562 +- 0.0059 | K=8 RL 0.1105 +- 0.0122
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
CKPT_DIR=${1:?checkpoint dir}
TAG=${2:?checkpoint tag, e.g. mean_k8_fulladv}
IFS=, read -r -a SEEDS <<< "${3:?comma-separated seeds}"
PREFIX=${4:-lcb_${TAG}}

if [ "$(strings $BIN 2>/dev/null | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build (0 cudarc symbols) — rebuild with --features cuda." ; exit 1
fi

gen_seed() { # $1=seed $2=gpubase -> 4 slices in parallel
  local s=$1 g=$2
  local OUT="scratch-7b-sft/${PREFIX}_s${s}"; mkdir -p "$OUT"
  local CKPT="$CKPT_DIR/${TAG}_seed${s}_final.safetensors"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; return 1; }
  for off in 640 670 700 730; do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 --checkpoint "$CKPT" --offset $off --n-problems 30 --passk 5 \
      --max-new-tokens 768 --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
}

n=${#SEEDS[@]}; idx=0
while [ $idx -lt $n ]; do
  a=${SEEDS[$idx]}; b=${SEEDS[$((idx+1))]}
  echo "=== $TAG LCB gen batch: $a (GPU0-3)${b:+ + $b (GPU4-7)} ==="
  gen_seed "$a" 0
  [ -n "$b" ] && gen_seed "$b" 4
  wait
  idx=$((idx + 2))
done
echo "=== all $TAG LCB generation done; scoring ==="

score() {
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
  echo "############ $TAG seed $s ############"; score "scratch-7b-sft/${PREFIX}_s${s}"
done
echo "=== LCB_ARM_AGG_COMPLETE ($TAG) ==="
