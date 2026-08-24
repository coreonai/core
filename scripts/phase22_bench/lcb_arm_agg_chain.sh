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
# Reference (6 seeds each, post-cutoff n=92 aggregate pass@1):
#   base 0.0413 | full-set SFT 0.0562 +- 0.0059
#   K=2 0.0841 | K=4 0.0982 | K=8 0.1105 | K=16 0.1293 | K=32 0.1279
#   K=16 is the saturation point (K=32-K=16 = -0.0014, t=-0.17, 3/6 seeds).
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
CKPT_DIR=${1:?checkpoint dir}
TAG=${2:?checkpoint tag, e.g. mean_k8_fulladv}
IFS=, read -r -a SEEDS <<< "${3:?comma-separated seeds}"
PREFIX=${4:-lcb_${TAG}}
# GPUs usable for generation, comma-separated. Each seed needs 4 (one per
# slice). Default 0-7 = two seeds in flight. Narrow it when a card is held by
# an unrelated process: F32 7B needs ~31 GB, so a card with e.g. 25 GB of
# someone else's vLLM on it will OOM the 4th slice and silently truncate a
# seed's problem set (that happened; hence the completeness check below).
IFS=, read -r -a GPUS <<< "${5:-0,1,2,3,4,5,6,7}"
SEEDS_AT_ONCE=$(( ${#GPUS[@]} / 4 ))
[ "$SEEDS_AT_ONCE" -lt 1 ] && { echo "⚠ need at least 4 GPUs, got ${#GPUS[@]}"; exit 1; }

if [ "$(strings $BIN 2>/dev/null | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build (0 cudarc symbols) — rebuild with --features cuda." ; exit 1
fi

gen_seed() { # $1=seed $2=index into GPUS of this seed's first card
  local s=$1 gi=$2
  local OUT="scratch-7b-sft/${PREFIX}_s${s}"; mkdir -p "$OUT"
  local CKPT="$CKPT_DIR/${TAG}_seed${s}_final.safetensors"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; return 1; }
  for off in 640 670 700 730; do
    local g=${GPUS[$gi]}; gi=$((gi + 1))
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 --checkpoint "$CKPT" --offset $off --n-problems 30 --passk 5 \
      --max-new-tokens 768 --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
  done
}

n=${#SEEDS[@]}; idx=0
while [ $idx -lt $n ]; do
  batch=()
  for j in $(seq 0 $((SEEDS_AT_ONCE - 1))); do
    [ $((idx + j)) -lt $n ] && batch+=("${SEEDS[$((idx + j))]}")
  done
  echo "=== $TAG LCB gen batch: ${batch[*]} on GPUs ${GPUS[*]} ==="
  k=0
  for sd in "${batch[@]}"; do gen_seed "$sd" $((k * 4)); k=$((k + 1)); done
  wait
  idx=$((idx + SEEDS_AT_ONCE))
done
echo "=== all $TAG LCB generation done; scoring ==="

# A slice that OOMs leaves no JSON, and `gens_all.json` is built from whatever
# is present — so a seed silently ends up scored on 90 problems instead of 120
# while its number still looks ordinary. (Hit for real: 3 seeds lost slice 730
# to an external process holding 25 GB on one card.) Refuse to score a seed
# whose slice set is incomplete.
EXPECTED_SLICES=4
score() {
  local OUT=$1
  local got; got=$(ls "$OUT"/slice_*.json 2>/dev/null | wc -l)
  if [ "$got" -ne "$EXPECTED_SLICES" ]; then
    echo "  ⚠ INCOMPLETE: $got/$EXPECTED_SLICES slices — NOT scored (would use a different problem set)"
    for off in 640 670 700 730; do
      [ -f "$OUT/slice_$off.json" ] || echo "      missing $off: $(sed -r 's/\x1b\[[0-9;]*m//g' "$OUT/gen_$off.log" 2>/dev/null | grep -m1 -iE 'error|out of memory' || echo 'no log')"
    done
    return 1
  fi
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
