#!/usr/bin/env python3
"""Phase 22 §6.5 — BigCodeBench Docker-scoring smoke.

Build a tiny known-correct samples file from the canonical solutions so we can
confirm the sample FORMAT and the calibrated/uncalibrated convention BEFORE
trusting any base number. Emits two variants for the first few Hard tasks:

  completion : solution = canonical_solution (the body only) — for --calibrated
  full       : solution = complete_prompt + canonical_solution (self-contained)

Whichever scores ~100% under the Docker harness is the convention our
Rust-generated (completion-body) samples must match.
"""
import json
import sys
from huggingface_hub import hf_hub_download
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
p = hf_hub_download(repo_id="bigcode/bigcodebench-hard", repo_type="dataset",
                    filename="data/v0.1.4-00000-of-00001.parquet")
df = pd.read_parquet(p).head(N)

with open("scratch-7b-sft/bcb_hard_base/smoke_completion.jsonl", "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({"task_id": r["task_id"],
                            "solution": r["canonical_solution"]}) + "\n")
with open("scratch-7b-sft/bcb_hard_base/smoke_full.jsonl", "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({"task_id": r["task_id"],
                            "solution": r["complete_prompt"] + r["canonical_solution"]}) + "\n")
print(f"wrote smoke_completion.jsonl + smoke_full.jsonl ({N} tasks:",
      list(df["task_id"]), ")")
