"""Phase 17 S3 — load MBPP into the {name, prompt, suffix} shape.

MBPP is a different format from HumanEval:
  text:       natural-language task description
  code:       canonical implementation (we parse signature from this)
  test_list:  list of assert statements
  task_id:    integer

We synthesize a HumanEval-style prompt by:
  1. Extracting function signature from `code` via ast
  2. Emitting `<imports>\\n<sig>:\\n    \"\"\"<text>\"\"\"\\n`
  3. Suffix = newline-joined test_list

We use task_id 11-110 (100 tasks) to skip the standard MBPP few-shot
prompt examples (task_id 1-10) and keep wallclock manageable.
"""

import ast
import json
import re
from pathlib import Path

MBPP_JSONL = Path(__file__).parent.parent.parent / "data" / "mbpp" / "mbpp.jsonl"


def parse_signature(code):
    """Return (function_name, signature_line) of first top-level def."""
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = ", ".join(a.arg for a in node.args.args)
            return node.name, f"def {node.name}({args}):"
    return None


def detect_imports(code):
    """Detect imports referenced in the canonical code; we re-emit the
    same imports in the prompt so the model has the right context."""
    imports = []
    if "import math" in code:
        imports.append("import math")
    if "from typing" in code or re.search(r"\b(List|Dict|Set|Tuple|Optional)\b", code):
        imports.append("from typing import *")
    if "import re" in code:
        imports.append("import re")
    if "import heapq" in code:
        imports.append("import heapq")
    if "import collections" in code or "Counter(" in code or "defaultdict(" in code:
        imports.append("import collections")
    if "from collections" in code:
        imports.append("from collections import *")
    if "import itertools" in code or "itertools." in code:
        imports.append("import itertools")
    if "import functools" in code or "reduce(" in code:
        imports.append("import functools")
    return imports


def load_mbpp(start_id=11, end_id=111, max_tasks=100):
    if not MBPP_JSONL.exists():
        raise FileNotFoundError(
            f"{MBPP_JSONL} not found. "
            f"Run: curl -fsSL https://raw.githubusercontent.com/"
            f"google-research/google-research/master/mbpp/mbpp.jsonl "
            f"-o {MBPP_JSONL}"
        )
    with open(MBPP_JSONL) as f:
        all_tasks = [json.loads(line) for line in f]
    selected = [t for t in all_tasks if start_id <= t["task_id"] < end_id]
    return selected[:max_tasks]


def to_challenges(tasks):
    challenges = []
    skipped = 0
    for t in tasks:
        sig_info = parse_signature(t["code"])
        if sig_info is None:
            skipped += 1
            continue
        func_name, sig = sig_info
        imports = detect_imports(t["code"])
        prelude = "\n".join(imports) + "\n\n" if imports else ""
        # Single-line description as docstring (some MBPP descriptions
        # have newlines; collapse for prompt cleanliness)
        desc = t["text"].replace("\n", " ").strip()
        prompt = f'{prelude}{sig}\n    """{desc}"""\n'
        # Suffix: also re-emit imports + tests (so verify can call back
        # the function with imports already loaded)
        suffix_imports = "\n".join(imports) + "\n" if imports else ""
        suffix = "\n" + suffix_imports + "\n".join(t["test_list"]) + "\n"
        challenges.append({
            "name": f"mbpp_{t['task_id']}",
            "prompt": prompt,
            "suffix": suffix,
            "entry_point": func_name,
        })
    if skipped:
        print(f"[problems.py] skipped {skipped} tasks with unparseable code")
    return challenges


CHALLENGES = to_challenges(load_mbpp())


if __name__ == "__main__":
    print(f"loaded {len(CHALLENGES)} MBPP challenges (task_id 11-110, parsed)")
    for c in CHALLENGES[:3]:
        print(f"\n--- {c['name']} (entry={c['entry_point']}) ---")
        print(f"prompt:")
        print(c['prompt'])
        print(f"suffix[:200]: {c['suffix'][:200]}")
