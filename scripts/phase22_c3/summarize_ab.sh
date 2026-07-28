#!/bin/bash
# Phase 22 follow-up C3 — table view of the A/B logs.
#
# Prints, per run, the per-step pass count / mean completion length /
# optimizer updates, plus the cumulative update count the *sampler* saw at
# that step (updates applied before the step's generation). That last column
# is the one that matters: the original Stage E collapse happened at 1024
# cumulative updates behind a SINGLE sync.
set -e
cd /raid/users/paul/workLLM
OUT=${1:-scratch-7b-sft/c3_ab}

for f in "$OUT"/*.log; do
  echo "=== $(basename "$f" .log)"
  printf '%6s %8s %10s %6s %12s\n' step pass comp_len upd cum_upd_seen
  sed -r 's/\x1b\[[0-9;]*m//g' "$f" \
    | awk '
      # `rl_steps = 6` in the header line also contains "rl_step" — require
      # the step number to follow.
      /rl_step [0-9]/ {
        step=""; pass=""; clen=""; upd=""
        for (i = 1; i <= NF; i++) {
          if ($i == "rl_step") step = $(i+1)
          if ($i == "pass")     pass = $(i+2)
          if ($i == "comp_len") clen = $(i+2)
          if ($i == "upd")      upd  = $(i+2)
        }
        printf "%6s %8s %10s %6s %12d\n", step, pass, clen, upd, cum
        cum += upd
      }'
  sed -r 's/\x1b\[[0-9;]*m//g' "$f" | grep -E "OUT_OF_MEMORY|panicked|^Error" | head -2
done
