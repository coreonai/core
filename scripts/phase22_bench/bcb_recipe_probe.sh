#!/bin/bash
# Phase 22 §6.5 — BigCodeBench recipe CEILING PROBE (Complete / Hard).
# Measures the self-improve recipes at the metric where the signal lives:
# AGGREGATE pass@1 (temp 0.8, passk 5, F32) — NOT greedy (lesson #11: SFT/RL
# sharpen the sampling distribution; greedy hides the gain). Also yields pass@5.
# Arms: base 7B, full-set SFT (3 seeds), K=8 RL (3 seeds). Same slice/metric as
# the base 0.169 leaderboard run, but aggregate rather than greedy.
# Generate in Rust (8-GPU) -> merge to {task_id, solution} x5 -> Docker score.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
JSONL=data/bigcodebench/BigCodeBench-Hard.jsonl
IMG=bigcodebench/bigcodebench-evaluate:latest
N=$(wc -l < "$JSONL")     # 148
PER=19                    # 8 slices
SEEDS=(42 100 200)

# arm spec: "label|checkpoint"  (empty checkpoint = base)
ARMS=("base|")
for s in "${SEEDS[@]}"; do ARMS+=("sft_s${s}|scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors"); done
for s in "${SEEDS[@]}"; do ARMS+=("k8_s${s}|scratch-7b-sft/rlvar_k8/mean_k8_seed${s}_final.safetensors"); done

gen_arm() { # $1=label $2=ckpt -> aggregate samples (passk 5, temp 0.8, F32)
  local label=$1 ckpt=$2
  local OUT="scratch-7b-sft/bcb_agg_${label}"; mkdir -p "$OUT"
  local ck=(); [ -n "$ckpt" ] && ck=(--checkpoint "$ckpt")
  [ -n "$ckpt" ] && [ ! -f "$ckpt" ] && { echo "⚠ missing $ckpt"; return 1; }
  local g=0
  for off in $(seq 0 $PER $((N-1))); do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
      --jsonl "$JSONL" --split complete --dtype f32 "${ck[@]}" \
      --offset $off --n-problems $PER --passk 5 --max-new-tokens 1024 \
      --dump "$OUT/slice_$off.jsonl" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
  wait
  cat "$OUT"/slice_*.jsonl > "$OUT/samples.jsonl"
  echo "  $label: $(wc -l < "$OUT/samples.jsonl") samples (expect $((N*5)))"
}

score_arm() { # $1=label -> pass@1 (aggregate) + pass@5 via Docker
  local label=$1
  local OUT="scratch-7b-sft/bcb_agg_${label}"; mkdir -p "$OUT/.dockerhome"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/app/.dockerhome \
    -v "$(cd "$OUT" && pwd)":/app -w /app "$IMG" \
    complete hard --samples samples.jsonl --execution local --calibrated True \
    > "$OUT/score.log" 2>&1
  python3 -c "
import json
d=json.load(open('$OUT/samples_pass_at_k.json'))
print('  $label: pass@1=%.4f pass@5=%s gt=%.3f' % (d.get('pass@1',-1), d.get('pass@5','NA'), d.get('gt_pass_rate',-1)))
" 2>/dev/null || echo "  $label: SCORE PARSE FAIL (see $OUT/score.log)"
}

for spec in "${ARMS[@]}"; do
  label=${spec%%|*}; ckpt=${spec#*|}
  echo "=== ARM $label — generate (aggregate passk5 F32) ==="
  gen_arm "$label" "$ckpt" || continue
  echo "=== ARM $label — score (Docker complete/hard calibrated) ==="
  score_arm "$label"
done

echo "=== CEILING PROBE SUMMARY (aggregate pass@1, base 0.169 greedy for context) ==="
python3 - <<'PY'
import json, glob, statistics as st, os
def pk(label):
    f=f"scratch-7b-sft/bcb_agg_{label}/samples_pass_at_k.json"
    if not os.path.exists(f): return None
    d=json.load(open(f)); return (d.get('pass@1'), d.get('pass@5'), d.get('gt_pass_rate'))
seeds=[42,100,200]
b=pk('base')
print('base   aggregate: pass@1=%.4f pass@5=%.4f gt=%.3f' % b if b else 'base: NA')
for arm in ('sft','k8'):
    vs=[pk(f'{arm}_s{s}') for s in seeds]; vs=[v for v in vs if v]
    if not vs: print(f'{arm}: NA'); continue
    p1=[v[0] for v in vs]; p5=[v[1] for v in vs]
    m1=st.mean(p1); s1=st.stdev(p1) if len(p1)>1 else 0.0
    m5=st.mean(p5); s5=st.stdev(p5) if len(p5)>1 else 0.0
    d=m1-(b[0] if b else 0)
    print(f'{arm:4s} {len(vs)}-seed: pass@1={m1:.4f}±{s1:.4f}  pass@5={m5:.4f}±{s5:.4f}  Δpass@1 vs base={d:+.4f}  per-seed p@1={[round(x,4) for x in p1]}')
PY
echo "=== BCB_PROBE_COMPLETE ==="
