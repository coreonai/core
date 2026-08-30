#!/usr/bin/env bash
# Chained after the widened 8-family self-improve run: measure all three
# axes on the FINAL checkpoint, on the same ruler as every prior number.
#
#   in-domain  families 3,5   — the targets (were 0.000 -> 1.000 when
#                               harvested alone)
#   retention  families 0,1,2,4,7 — were ~0.988 before any loop, 0.806
#                               after the 3+5-only run
#   transfer   --novel        — divisors/Fibonacci/Collatz, never in any
#                               harvest pool. 12/12 emitted / 4/12 correct
#                               before the loop, 4/12 / 1/12 after the
#                               3+5-only run
#
# Waits by PID, not `pgrep -f`: a pgrep pattern matching this script's own
# command line counts itself and never reaches zero. That deadlock has
# burned this project twice, once idling 8 GPUs for three hours.
set -u
cd /raid/users/paul/workLLM

MAIN_PID="${1:?usage: $0 <main-run-pid>}"
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 60; done
echo "main run $MAIN_PID exited; measuring"

OUT=scratch-7b-sft/p23_si_all8
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/*/ | head -1)

# Highest-numbered round checkpoint = the final model.
CK=$(ls -1 "$OUT"/r0_merged.r*.safetensors 2>/dev/null | sort -V | tail -1)
if [ -z "$CK" ]; then echo "FATAL: no checkpoint under $OUT"; exit 1; fi
echo "final checkpoint = $CK"

DIR=scratch-7b-sft/p23_all8_final_dir
mkdir -p "$DIR"
ln -sf "$(realpath "$CK")" "$DIR/model.safetensors"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json vocab.json merges.txt; do
  [ -f "$SNAP/$f" ] && ln -sf "$SNAP/$f" "$DIR/$f"
done

# in-domain targets — n range matches every earlier family-3/5 number
CUDA_VISIBLE_DEVICES=2 ./target/release/examples/phase23_tooluse_self_improve \
  --init-dir "$DIR" --baseline --families 3,5 --n-lo 12 --n-hi 60 \
  --out-dir scratch-7b-sft/p23_all8_targets \
  > scratch-7b-sft/p23_all8_targets.log 2>&1 &
P1=$!

# retention — n range matches the earlier 0.988/0.806 measurements
CUDA_VISIBLE_DEVICES=3 ./target/release/examples/phase23_tooluse_self_improve \
  --init-dir "$DIR" --baseline --families 0,1,2,4,7 --n-lo 12 --n-hi 43 \
  --out-dir scratch-7b-sft/p23_all8_retain \
  > scratch-7b-sft/p23_all8_retain.log 2>&1 &
P2=$!

# transfer — same flags as both earlier --novel runs
CUDA_VISIBLE_DEVICES=4 ./target/release/examples/phase23_python_tool_7b \
  --checkpoint "$CK" --no-shots --novel --n-problems 12 \
  --max-new-tokens 128 --show 4 --show-chars 220 \
  > scratch-7b-sft/p23_all8_transfer.log 2>&1 &
P3=$!

wait $P1 $P2 $P3
echo "ALL8_MEASUREMENTS_DONE"
