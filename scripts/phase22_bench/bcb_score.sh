#!/bin/bash
# Phase 22 §6.5 — BigCodeBench Docker scoring (Complete / Hard).
# Score a samples JSONL {task_id, solution} (solution = completion body) with
# the official harness inside the sandbox image. calibrated=True prepends the
# prompt automatically (smoke-confirmed: canonical solutions -> pass@1 1.000).
# Scoring is CPU-bound and does not touch the GPUs.
#
# Usage: bcb_score.sh <samples.jsonl>   (default: base Hard samples)
set +e
cd /raid/users/paul/workLLM
SAMPLES=${1:-scratch-7b-sft/bcb_hard_base/bcb_hard_base_samples.jsonl}
DIR=$(cd "$(dirname "$SAMPLES")" && pwd)
FILE=$(basename "$SAMPLES")
mkdir -p "$DIR/.dockerhome"
IMG=bigcodebench/bigcodebench-evaluate:latest

echo "=== scoring $FILE (complete/hard, calibrated, execution local) ==="
# --user so results files are owned by us; HOME writable for the HF dataset cache.
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/app/.dockerhome \
  -v "$DIR":/app -w /app "$IMG" \
  complete hard --samples "$FILE" --execution local --calibrated True 2>&1 \
  | tee "$DIR/${FILE%.jsonl}_score.log" | tail -20
echo "=== result files ==="; ls -la "$DIR/${FILE%.jsonl}"_*.json 2>/dev/null
echo "=== BCB_SCORE_COMPLETE ==="
