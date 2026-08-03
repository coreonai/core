#!/usr/bin/env python3
"""Phase 22 §6.5 — export BigCodeBench problems to the JSONL the Rust dumper reads.

Writes {task_id, complete_prompt, instruct_prompt} lines to
data/bigcodebench/BigCodeBench-<Subset>.jsonl. Pin v0.1.4 to match the bundled
dataset version in bigcodebench/bigcodebench-evaluate:latest (bcb 0.2.4), so the
task_ids/tests line up at scoring time.

Usage: bcb_export_problems.py [hard|full]   (default hard, 148 tasks)
"""
import json
import sys
from huggingface_hub import hf_hub_download
import pandas as pd

VERSION = "v0.1.4"
subset = (sys.argv[1] if len(sys.argv) > 1 else "hard").lower()
repo = "bigcode/bigcodebench-hard" if subset == "hard" else "bigcode/bigcodebench"
out = f"data/bigcodebench/BigCodeBench-{'Hard' if subset == 'hard' else 'Full'}.jsonl"

p = hf_hub_download(repo_id=repo, repo_type="dataset",
                    filename=f"data/{VERSION}-00000-of-00001.parquet")
df = pd.read_parquet(p)
import os
os.makedirs("data/bigcodebench", exist_ok=True)
with open(out, "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({
            "task_id": r["task_id"],
            "complete_prompt": r["complete_prompt"],
            "instruct_prompt": r.get("instruct_prompt", ""),
        }) + "\n")
print(f"wrote {len(df)} tasks ({repo} {VERSION}) -> {out}")
