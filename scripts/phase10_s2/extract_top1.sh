#!/bin/bash
# Phase 10 S2: pull `top-1 softmax mass` lines from the training logs
# and label by variant. Sources: /tmp/p10s2_lambda.log, p10s2_k.log,
# p10s2_ema_python.log (EMA block).

set -e
cd /raid/users/paul/workLLM

echo "=== top-1 softmax mass per K8 variant ==="
echo

# λ sweep + S1 baseline + S1 jepa01 (those came from earlier S1 logs).
# We only have logs for what *this* session's runs produced; for S1
# values, recall: baseline = 0.146, λ=0.1 = 0.097.

# Helper: read the previous block's heading from the log to label.
extract_blocks() {
  local logfile="$1"
  awk '
    # Match variant headings like "=== λ=0.01, k=8 ===" or "=== K8 EMA λ=0.1 k=8 decay=0.99 ===",
    # but NOT generic section headings.
    /^=== / && !/Phase 10 S1/ && !/sample/ && !/Python/ && !/end/ && !/done/ {
      label = $0
      sub(/^=== /, "", label); sub(/ ===.*/, "", label)
    }
    /top-1 softmax mass:/ {
      split($0, p, ":")
      val = p[2]
      gsub(/^[[:space:]]+/, "", val)
      gsub(/[[:space:]]+\(.*$/, "", val)
      printf("%-40s top1=%s\n", label, val)
    }
  ' "$logfile"
}

[ -f /tmp/p10s2_lambda.log ] && extract_blocks /tmp/p10s2_lambda.log
[ -f /tmp/p10s2_k.log ]      && extract_blocks /tmp/p10s2_k.log
[ -f /tmp/p10s2_ema_python.log ] && extract_blocks /tmp/p10s2_ema_python.log
echo
echo "(For S1 reference: baseline λ=0 top1=0.1456, S1 λ=0.1 k=8 top1=0.0971)"
