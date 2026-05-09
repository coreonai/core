"""Phase 15 S1 — load HumanEval (164 problems) into the
{name, prompt, suffix} shape Phase 14 used.

HumanEval task structure:
  prompt           : signature + docstring (model continues from here)
  entry_point      : function name to test
  canonical_solution: reference (we don't use)
  test             : defines `check(candidate)`

Verifier composition: prompt + completion + "\n\n" + test +
"\ncheck(<entry_point>)\n".
"""

import json
from pathlib import Path


HUMAN_EVAL_JSONL = Path(__file__).parent.parent.parent / "data" / "humaneval" / "HumanEval.jsonl"


def load_humaneval():
    if not HUMAN_EVAL_JSONL.exists():
        raise FileNotFoundError(
            f"{HUMAN_EVAL_JSONL} not found. "
            f"Run: curl -fsSL https://raw.githubusercontent.com/openai/human-eval/"
            f"master/data/HumanEval.jsonl.gz -o {HUMAN_EVAL_JSONL}.gz "
            f"&& gunzip {HUMAN_EVAL_JSONL}.gz"
        )
    with open(HUMAN_EVAL_JSONL) as f:
        return [json.loads(line) for line in f]


def to_challenges(tasks):
    """Convert HumanEval tasks to {name, prompt, suffix} shape."""
    challenges = []
    for t in tasks:
        suffix = "\n\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
        challenges.append({
            "name": t["task_id"].replace("HumanEval/", "he_"),
            "prompt": t["prompt"],
            "suffix": suffix,
            "entry_point": t["entry_point"],
        })
    return challenges


CHALLENGES = to_challenges(load_humaneval())


if __name__ == "__main__":
    print(f"loaded {len(CHALLENGES)} HumanEval challenges")
    c = CHALLENGES[0]
    print(f"--- {c['name']} (entry={c['entry_point']}) ---")
    print(f"prompt[:200]:\n{c['prompt'][:200]}")
    print(f"suffix[:200]:\n{c['suffix'][:200]}")
