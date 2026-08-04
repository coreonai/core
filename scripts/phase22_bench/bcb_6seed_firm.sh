#!/bin/bash
# Phase 22 §6.5 — firm the BigCodeBench ceiling probe from 3 to 6 seeds.
# WAITS for the GPUs to free (an ollama systemd service currently holds ~7GB on
# all 8 cards; F32 7B needs ~31GB so we must not run on top of it), then
# generates + Docker-scores the 3 new seeds (300,400,500) for BOTH recipes at
# aggregate pass@1 (temp 0.8, passk 5, F32), with per-slice auto-retry to absorb
# the intermittent slice_19 OOM. Finally re-aggregates all 6 seeds (42,100,200
# already done) for SFT and K=8 RL vs base 0.1459.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
JSONL=data/bigcodebench/BigCodeBench-Hard.jsonl
IMG=bigcodebench/bigcodebench-evaluate:latest
N=$(wc -l < "$JSONL")     # 148
PER=19
NEW_SEEDS=(300 400 500)

expect_for() { [ "$1" -lt 133 ] && echo 95 || echo 75; }   # last slice = 15 tasks x5

# arm spec: label|checkpoint
ARMS=()
for s in "${NEW_SEEDS[@]}"; do ARMS+=("sft_s${s}|scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors"); done
for s in "${NEW_SEEDS[@]}"; do ARMS+=("k8_s${s}|scratch-7b-sft/rlvar_k8/mean_k8_seed${s}_final.safetensors"); done

# ---- 1) wait for the GPUs to free (ollama releases memory when idle) ----
echo "=== waiting for GPUs to free (ollama to release) ==="
for i in $(seq 1 240); do   # up to ~4h
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  llama=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c llama-server)
  mine=$(pgrep -f phase22_dump_completions | grep -v pgrep | wc -l)
  if [ "$maxmem" -lt 2000 ] && [ "$llama" -eq 0 ] && [ "$mine" -eq 0 ]; then
    echo "GPUS_FREE after ~$((i)) min (maxmem=${maxmem}MiB)"; break
  fi
  [ $((i % 5)) -eq 0 ] && echo "  still busy: maxmem=${maxmem}MiB llama=${llama} (min $i)"
  sleep 60
done

gen_slice() { # $1=gpu $2=off $3=out $4=ckpt
  local g=$1 off=$2 out=$3 ckpt=$4
  local ck=(); [ -n "$ckpt" ] && ck=(--checkpoint "$ckpt")
  CUDA_VISIBLE_DEVICES=$g $BIN --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
    --jsonl "$JSONL" --split complete --dtype f32 "${ck[@]}" \
    --offset $off --n-problems $PER --passk 5 --max-new-tokens 1024 \
    --dump "$out/slice_$off.jsonl" > "$out/gen_$off.log" 2>&1
}

gen_arm() { # $1=label $2=ckpt  -> generate 8 slices (8-GPU) + retry incomplete ones alone
  local label=$1 ckpt=$2
  local OUT="scratch-7b-sft/bcb_agg_${label}"; mkdir -p "$OUT"
  [ -n "$ckpt" ] && [ ! -f "$ckpt" ] && { echo "  ⚠ missing $ckpt"; return 1; }
  local g=0
  for off in $(seq 0 $PER $((N-1))); do gen_slice $g $off "$OUT" "$ckpt" & g=$((g+1)); done
  wait
  # retry incomplete slices, one at a time on GPU 0 (no contention -> full 40GB)
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

score_arm() { # $1=label
  local OUT="scratch-7b-sft/bcb_agg_${1}"; mkdir -p "$OUT/.dockerhome"
  local ns=$(wc -l < "$OUT/samples.jsonl")
  [ "$ns" -ne $((N*5)) ] && { echo "  ⚠ $1 incomplete ($ns) — skip score"; return 1; }
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/app/.dockerhome \
    -v "$(cd "$OUT" && pwd)":/app -w /app "$IMG" \
    complete hard --samples samples.jsonl --execution local --calibrated True \
    > "$OUT/score.log" 2>&1
  python3 -c "
import json; d=json.load(open('$OUT/samples_pass_at_k.json'))
print('  $1: pass@1=%.4f pass@5=%s gt=%.3f'%(d.get('pass@1',-1),d.get('pass@5','NA'),d.get('gt_pass_rate',-1)))
" 2>/dev/null || echo "  $1: SCORE PARSE FAIL"
}

# ---- 2) generate + score the 6 new arms ----
for spec in "${ARMS[@]}"; do
  label=${spec%%|*}; ckpt=${spec#*|}
  echo "=== ARM $label — generate (aggregate passk5 F32) ==="; gen_arm "$label" "$ckpt" || continue
  echo "=== ARM $label — score ==="; score_arm "$label"
done

# ---- 3) 6-seed summary (existing 42,100,200 + new 300,400,500) ----
echo "=== 6-SEED CEILING SUMMARY (aggregate pass@1 vs base 0.1459) ==="
python3 - <<'PY'
import json, os, statistics as st
def pk(l):
    f=f"scratch-7b-sft/bcb_agg_{l}/samples_pass_at_k.json"
    return json.load(open(f)) if os.path.exists(f) else None
b=pk('base'); print("base aggregate: pass@1=%.4f pass@5=%.4f"%(b['pass@1'],b['pass@5']))
seeds=[42,100,200,300,400,500]
for arm in ('sft','k8'):
    vs=[(s,pk(f'{arm}_s{s}')) for s in seeds]; vs=[(s,v) for s,v in vs if v]
    p1=[v['pass@1'] for _,v in vs]; p5=[v['pass@5'] for _,v in vs]
    m1,s1=st.mean(p1),(st.stdev(p1) if len(p1)>1 else 0)
    m5,s5=st.mean(p5),(st.stdev(p5) if len(p5)>1 else 0)
    print(f"{arm:4s} {len(vs)}-seed: pass@1={m1:.4f}±{s1:.4f} pass@5={m5:.4f}±{s5:.4f} Δp@1={m1-b['pass@1']:+.4f} seeds={[s for s,_ in vs]} p@1={[round(x,4) for x in p1]}")
ms=[pk(f'sft_s{s}') for s in seeds]; mk=[pk(f'k8_s{s}') for s in seeds]
ms=[v['pass@1'] for v in ms if v]; mk=[v['pass@1'] for v in mk if v]
if len(ms)>1 and len(mk)>1:
    d=st.mean(mk)-st.mean(ms); pooled=(st.stdev(mk)**2+st.stdev(ms)**2)**0.5
    print(f"K8 vs SFT pass@1: Δ={d:+.4f} ({d/pooled:+.2f}σ_pooled) all_k8>all_sft={min(mk)>max(ms)}")
PY
echo "=== BCB_6SEED_COMPLETE ==="
