#!/bin/bash
# Phase 22 §6.5 — recover the 3 ceiling-probe arms whose slice_19 hit CUDA OOM
# (checkpoint-load + F32 passk5 contention on a shared GPU). Regenerate slice_19
# ALONE on a dedicated GPU (full 40GB, no contention), re-merge, re-score.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
JSONL=data/bigcodebench/BigCodeBench-Hard.jsonl
IMG=bigcodebench/bigcodebench-evaluate:latest

# arm -> checkpoint
declare -A CK=(
  [sft_s200]=scratch-7b-sft/hep_out_s200/r0_merged.r2.safetensors
  [k8_s42]=scratch-7b-sft/rlvar_k8/mean_k8_seed42_final.safetensors
  [k8_s100]=scratch-7b-sft/rlvar_k8/mean_k8_seed100_final.safetensors
)
ARMS=(sft_s200 k8_s42 k8_s100)

# regenerate slice_19 for each arm, one per dedicated GPU, concurrently.
g=0
for label in "${ARMS[@]}"; do
  d="scratch-7b-sft/bcb_agg_${label}"
  CUDA_VISIBLE_DEVICES=$g $BIN --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
    --jsonl "$JSONL" --split complete --dtype f32 --checkpoint "${CK[$label]}" \
    --offset 19 --n-problems 19 --passk 5 --max-new-tokens 1024 \
    --dump "$d/slice_19.jsonl" > "$d/gen_19_recover.log" 2>&1 &
  g=$((g + 1))
done
wait
echo "=== slice_19 regenerated; re-merge + verify ==="
for label in "${ARMS[@]}"; do
  d="scratch-7b-sft/bcb_agg_${label}"
  cat "$d"/slice_*.jsonl > "$d/samples.jsonl"
  echo "  $label: slice_19=$(wc -l < "$d/slice_19.jsonl") samples=$(wc -l < "$d/samples.jsonl") (expect 740)"
done

echo "=== re-score the 3 recovered arms (Docker) ==="
for label in "${ARMS[@]}"; do
  d="scratch-7b-sft/bcb_agg_${label}"; mkdir -p "$d/.dockerhome"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/app/.dockerhome \
    -v "$(cd "$d" && pwd)":/app -w /app "$IMG" \
    complete hard --samples samples.jsonl --execution local --calibrated True \
    > "$d/score.log" 2>&1
  python3 -c "
import json
d=json.load(open('$d/samples_pass_at_k.json'))
print('  $label: pass@1=%.4f pass@5=%s gt=%.3f' % (d.get('pass@1',-1), d.get('pass@5','NA'), d.get('gt_pass_rate',-1)))
" 2>/dev/null || echo "  $label: SCORE PARSE FAIL"
done
echo "=== BCB_RECOVER_COMPLETE ==="
