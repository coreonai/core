#!/usr/bin/env python3
"""Phase 22 §6.5 — score Rust-generated LCB completions with the official eval core.

"Generate in Rust, score with the official harness." Reads our dumped
`[{question_id, code_list}]` (bench_export::write_lcb) and scores it with
lcb_runner's `codegen_metrics` (the same code-execution + pass@k the leaderboard
uses), optionally restricted to a contest-date window for the contamination
split.

Usage (light scoring venv):
  scratch-7b-sft/tools/lcb-venv/bin/python scripts/phase22_bench/lcb_score.py \
      --gens <path.json> --release release_v5 [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO = "/raid/users/paul/workLLM"
sys.path.insert(0, os.path.join(REPO, "scratch-7b-sft/tools/LiveCodeBench"))

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics  # noqa: E402
from lcb_runner.evaluation.pass_k_utils import extract_instance_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="[{question_id, code_list}] JSON")
    ap.add_argument("--release", default="release_v5")
    ap.add_argument("--start-date", default=None, help="keep problems on/after (post-cutoff)")
    ap.add_argument("--end-date", default=None, help="keep problems before (pre-cutoff)")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    with open(args.gens) as f:
        gens = json.load(f)
    by_id = {g["question_id"]: g["code_list"] for g in gens}
    print(f"[lcb-score] {len(by_id)} generated question_ids", flush=True)

    problems = load_code_generation_dataset(
        release_version=args.release, start_date=args.start_date, end_date=args.end_date
    )
    win = ""
    if args.start_date:
        win += f" start>={args.start_date}"
    if args.end_date:
        win += f" end<{args.end_date}"
    print(f"[lcb-score] {len(problems)} problems in window{win or ' (all)'}", flush=True)

    samples, generations, matched_ids = [], [], []
    for p in problems:
        qid = str(p.question_id)
        if qid not in by_id:
            continue
        samples.append(p.get_evaluation_sample())
        generations.append(by_id[qid])
        matched_ids.append(qid)
    print(f"[lcb-score] {len(samples)} problems matched (in-window AND generated)", flush=True)
    if not samples:
        print("[lcb-score] nothing to score in this window", flush=True)
        return

    res = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=args.procs)
    metrics = res[0] if isinstance(res, (list, tuple)) else res
    # per-problem pass (any of its generations passes all tests)
    graded = extract_instance_results(res[1]) if isinstance(res, (list, tuple)) and len(res) > 1 else None
    manual_p1 = None
    if graded is not None:
        flat = [any(g) for g in graded]
        manual_p1 = sum(flat) / max(1, len(flat))

    p1 = metrics.get("pass@1") if isinstance(metrics, dict) else None
    print("\n=== LCB RESULT ===")
    print(f"window: {win or 'all'}   matched problems: {len(samples)}")
    print(f"pass@1 (codegen_metrics): {p1}")
    if manual_p1 is not None:
        print(f"pass@1 (per-problem mean): {manual_p1:.4f}")
    print("==================")


if __name__ == "__main__":
    main()
