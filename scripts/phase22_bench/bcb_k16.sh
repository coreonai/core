#!/bin/bash
# Phase 22 §6.5 — BigCodeBench at the TUNED harvest point (K=16).
#
# The published BCB ceiling probe used K=8, which the transfer sweep has since
# shown is not the operating point: LCB post-cutoff rises log-linearly to K=16
# (+0.0148/doubling, 6/6 seeds, t=3.68) and is flat at K=32. K=16 is worth
# +0.088 over base on LCB vs K=8's +0.069, so the BCB number measured at K=8
# understates the recipe.
#
# Same ruler as bcb_6seed_firm.sh — Complete/Hard, aggregate pass@1, temp 0.8,
# passk 5, F32, max_new 1024, Docker scoring, calibrated. Only the checkpoints
# differ, so the result is directly comparable to the recorded
# base 0.1459 / SFT 0.155 / K=8 0.181.
#
# Usage: bcb_k16.sh [seeds]        default 42,100,200,300,400,500
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
JSONL=data/bigcodebench/BigCodeBench-Hard.jsonl
IMG=bigcodebench/bigcodebench-evaluate:latest
N=$(wc -l < "$JSONL")     # 148
PER=19
IFS=, read -r -a SEEDS <<< "${1:-42,100,200,300,400,500}"

[ "$(strings $BIN | grep -c cudarc)" -eq 0 ] && { echo "⚠ $BIN is a CPU build"; exit 1; }

# last slice covers 15 tasks, not 19
expect_for() { [ "$1" -lt 133 ] && echo 95 || echo 75; }

# F32 7B needs ~31 GB. External jobs (ollama, vLLM) turn up on this box
# intermittently and silently truncate a slice to an OOM, which would score a
# seed on fewer problems than the others. Wait for the cards rather than race.
echo "=== waiting for GPUs to free ==="
for i in $(seq 1 240); do
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  mine=$(pgrep -f phase22_dump_completions | grep -v pgrep | wc -l)
  if [ "$maxmem" -lt 2000 ] && [ "$mine" -eq 0 ]; then
    echo "GPUS_FREE after ${i} min (maxmem=${maxmem}MiB)"; break
  fi
  [ $((i % 5)) -eq 0 ] && echo "  still busy: maxmem=${maxmem}MiB (min $i)"
  sleep 60
done

gen_slice() { # $1=gpu $2=off $3=out $4=ckpt
  local g=$1 off=$2 out=$3 ckpt=$4
  CUDA_VISIBLE_DEVICES=$g $BIN --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
    --jsonl "$JSONL" --split complete --dtype f32 --checkpoint "$ckpt" \
    --offset $off --n-problems $PER --passk 5 --max-new-tokens 1024 \
    --dump "$out/slice_$off.jsonl" > "$out/gen_$off.log" 2>&1
}

gen_arm() { # $1=label $2=ckpt — 8 slices in parallel, then retry any short ones alone
  local label=$1 ckpt=$2
  local OUT="scratch-7b-sft/bcb_agg_${label}"; mkdir -p "$OUT"
  [ ! -f "$ckpt" ] && { echo "  ⚠ missing $ckpt"; return 1; }
  local g=0
  for off in $(seq 0 $PER $((N-1))); do gen_slice $g $off "$OUT" "$ckpt" & g=$((g+1)); done
  wait
  for round in 1 2; do
    local bad=0
    for off in $(seq 0 $PER $((N-1))); do
      local exp=$(expect_for $off)
      local got=$([ -f "$OUT/slice_$off.jsonl" ] && wc -l < "$OUT/slice_$off.jsonl" || echo 0)
      if [ "$got" -ne "$exp" ]; then
        echo "  [retry r$round] $label slice_$off: $got/$exp -> regen alone on GPU0"
        gen_slice 0 $off "$OUT" "$ckpt"; bad=$((bad+1))
      fi
    done
    [ "$bad" -eq 0 ] && break
  done
  cat "$OUT"/slice_*.jsonl > "$OUT/samples.jsonl"
  echo "  $label: $(wc -l < "$OUT/samples.jsonl") samples (expect $((N*5)))"
}

score_arm() { # $1=label — refuses to score a short sample set
  local OUT="scratch-7b-sft/bcb_agg_${1}"; mkdir -p "$OUT/.dockerhome"
  local ns=$(wc -l < "$OUT/samples.jsonl")
  [ "$ns" -ne $((N*5)) ] && { echo "  ⚠ $1 incomplete ($ns/$((N*5))) — NOT scored"; return 1; }
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/app/.dockerhome \
    -v "$(cd "$OUT" && pwd)":/app -w /app "$IMG" \
    complete hard --samples samples.jsonl --execution local --calibrated True \
    > "$OUT/score.log" 2>&1
  python3 -c "
import json; d=json.load(open('$OUT/samples_pass_at_k.json'))
print('  $1: pass@1=%.4f pass@5=%s'%(d.get('pass@1',-1),d.get('pass@5','NA')))
" 2>/dev/null || echo "  $1: SCORE PARSE FAIL"
}

for s in "${SEEDS[@]}"; do
  label="k16_s${s}"
  ckpt="scratch-7b-sft/rlvar_k16/mean_k16_seed${s}_final.safetensors"
  echo "=== ARM $label — generate ==="; gen_arm "$label" "$ckpt" || continue
  echo "=== ARM $label — score ==="; score_arm "$label"
done

echo "=== BCB CEILING, K=16 vs recorded arms (aggregate pass@1, Complete/Hard) ==="
python3 - <<'PY'
import json, os, statistics as st
def pk(l):
    f=f"scratch-7b-sft/bcb_agg_{l}/samples_pass_at_k.json"
    return json.load(open(f)) if os.path.exists(f) else None
b=pk('base')
print("base: pass@1=%.4f pass@5=%.4f" % (b['pass@1'], b['pass@5']))
seeds=[42,100,200,300,400,500]
rows={}
for arm in ('sft','k8','k16'):
    vs=[(s,pk(f'{arm}_s{s}')) for s in seeds]; vs=[(s,v) for s,v in vs if v]
    if not vs: continue
    p1=[v['pass@1'] for _,v in vs]; p5=[v['pass@5'] for _,v in vs]
    rows[arm]=p1
    m,sd=st.mean(p1),(st.stdev(p1) if len(p1)>1 else 0)
    print("%-4s %d-seed: pass@1=%.4f±%.4f  pass@5=%.4f  Δbase=%+.4f  %s"
          % (arm,len(vs),m,sd,st.mean(p5),m-b['pass@1'],[round(x,4) for x in p1]))
if 'k16' in rows and 'k8' in rows and len(rows['k16'])==len(rows['k8']):
    d=[a-c for a,c in zip(rows['k16'],rows['k8'])]
    m,sd=st.mean(d),st.stdev(d)
    t=m/(sd/len(d)**0.5) if sd else float('inf')
    print("paired k16-k8: %+.4f (sd %.4f, t=%.2f, df=%d, k16 ahead %d/%d)"
          % (m,sd,t,len(d)-1,sum(1 for x in d if x>0),len(d)))
PY
echo "=== BCB_K16_COMPLETE ==="
