#!/usr/bin/env python3
"""Phase 22 §6.5 — export LiveCodeBench problems for the Rust generation path.

Loads the official LCB code-generation dataset (via the vendored lcb_runner
benchmark loader) and writes one JSON object per problem with the fields the
Rust `BigCodeBenchDomain`-style prompt source needs to GENERATE completions:
  {question_id, question_content, starter_code, contest_date, platform, difficulty}

Scoring stays with the official eval core (codegen_metrics) which reloads the
full problem (with test cases) from the HF cache — this export is generation-only.

Run with the light scoring venv:
  scratch-7b-sft/tools/lcb-venv/bin/python scripts/phase22_bench/lcb_export_problems.py release_v5
"""
import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")

REPO = "/raid/users/paul/workLLM"
sys.path.insert(0, os.path.join(REPO, "scratch-7b-sft/tools/LiveCodeBench"))

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset  # noqa: E402


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "release_v5"
    out_dir = os.path.join(REPO, "data/livecodebench")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lcb_{version}.jsonl")

    print(f"[lcb-export] loading {version} (downloads + caches on first run) ...", flush=True)
    probs = load_code_generation_dataset(release_version=version)
    print(f"[lcb-export] {len(probs)} problems", flush=True)

    dates = []
    with open(out_path, "w") as f:
        for p in probs:
            date = str(getattr(p, "contest_date", ""))[:10]
            dates.append(date)
            row = {
                "question_id": str(getattr(p, "question_id", "")),
                "question_content": str(getattr(p, "question_content", "")),
                "starter_code": str(getattr(p, "starter_code", "") or ""),
                "contest_date": date,
                "platform": str(getattr(p, "platform", "")),
                "difficulty": str(getattr(p, "difficulty", "")),
            }
            f.write(json.dumps(row) + "\n")

    dates.sort()
    print(f"[lcb-export] wrote {len(probs)} -> {out_path}", flush=True)
    print(f"[lcb-export] contest_date range: {dates[0]} .. {dates[-1]}", flush=True)
    # date histogram by year-month, for choosing the cutoff split
    from collections import Counter
    ym = Counter(d[:7] for d in dates if d)
    print("[lcb-export] problems per year-month:", flush=True)
    for k in sorted(ym):
        print(f"    {k}: {ym[k]}", flush=True)


if __name__ == "__main__":
    main()
