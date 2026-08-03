#!/bin/bash
# Phase 22 §6.5 — firm the LCB generalization result to 6 seeds.
# Wait for full-set SFT training (seeds 100,300,400,500), then run LCB at the
# correct metric (aggregate pass@1, passk 5, temp 0.8, F32) for the 5 seeds not
# yet measured (100,200,300,400,500; seed 42 = lcb_agg_hep already done), score
# each (codegen aggregate pass@1, pre/post cutoff), and summarise 6-seed vs base
# 0.041 post-cutoff.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
NEW_TRAIN=(100 300 400 500)          # seeds being trained now
LCB_SEEDS=(100 200 300 400 500)      # seeds needing LCB agg (42 already done)

# 1) wait for the 4 new training checkpoints + no trainer procs.
for i in $(seq 1 300); do
  have=0
  for s in "${NEW_TRAIN[@]}"; do [ -f "scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors" ] && have=$((have+1)); done
  alive=$(pgrep -af phase22_he_mr_sft | grep -v pgrep | wc -l)
  if [ "$have" -ge 4 ] && [ "$alive" -eq 0 ]; then echo "TRAIN_DONE have=$have"; break; fi
  if [ "$alive" -eq 0 ] && [ "$i" -gt 5 ]; then echo "PROCS_EXITED have=$have"; break; fi
  sleep 120
done

gen_seed() { # $1=seed $2=gpubase  -> 4 slices, passk 5, F32, checkpoint
  local s=$1 g=$2
  local OUT="scratch-7b-sft/lcb_agg_s${s}"; mkdir -p "$OUT"
  local CKPT="scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; return 1; }
  for off in 640 670 700 730; do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 --checkpoint "$CKPT" --offset $off --n-problems 30 --passk 5 \
      --max-new-tokens 768 --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
}

# 2) generate LCB agg for the 5 seeds, 2 at a time (8 GPUs).
n=${#LCB_SEEDS[@]}
idx=0
while [ $idx -lt $n ]; do
  a=${LCB_SEEDS[$idx]}; b=${LCB_SEEDS[$((idx+1))]}
  echo "=== LCB gen batch: $a (GPU0-3)${b:+ + $b (GPU4-7)} ==="
  gen_seed "$a" 0
  [ -n "$b" ] && gen_seed "$b" 4
  wait
  idx=$((idx + 2))
done
echo "=== all LCB generation done; scoring ==="

# 3) score each seed (aggregate pass@1, pre/post) + collect post values.
score() { # $1=gensdir label
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
echo "############ seed 42 (already measured) ############"; score scratch-7b-sft/lcb_agg_hep
for s in "${LCB_SEEDS[@]}"; do
  echo "############ seed $s ############"; score "scratch-7b-sft/lcb_agg_s${s}"
done

# 4) 6-seed post-cutoff summary vs base.
echo "=== 6-SEED SUMMARY (post-cutoff aggregate pass@1) ==="
$V/python - <<PY
import json, glob, statistics as st, subprocess, re, os
V="$V"; CUT="$CUT"
def post(gens):
    out=subprocess.run([f"{V}/python","scripts/phase22_bench/lcb_score.py","--gens",gens,
                        "--release","release_v5","--start-date",CUT],capture_output=True,text=True)
    m=re.search(r"pass@1 \(codegen_metrics\): ([0-9.]+)", out.stdout)
    return float(m.group(1)) if m else None
dirs={42:"scratch-7b-sft/lcb_agg_hep"}
for s in [100,200,300,400,500]: dirs[s]=f"scratch-7b-sft/lcb_agg_s{s}"
vals={}
for s,d in dirs.items():
    g=os.path.join(d,"gens_all.json")
    if os.path.exists(g): vals[s]=post(g)
print("per-seed post-cutoff:", {k:round(v,4) for k,v in vals.items() if v is not None})
v=[x for x in vals.values() if x is not None]
if len(v)>=2:
    print(f"6-seed post-cutoff mean={st.mean(v):.4f} sigma={st.stdev(v):.4f}  (base=0.0413)")
    print(f"delta vs base = {st.mean(v)-0.0413:+.4f}")
PY
echo "=== LCB_6SEED_COMPLETE ==="
