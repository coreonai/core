"""Phase 15 S2 — split HumanEval 164 into k=3 specialist subsets by
signature-line heuristic.

Subsets (deterministic, inspectable):
- strings:     signature involves str / List[str] / Dict[str, ...]
- collections: any List/Dict/Set/Tuple type that isn't covered by strings
- numbers:     pure int/float/bool work; no string/collection types

Heuristic is intentionally simple — first match wins, in this order:
strings → collections → numbers. The split won't be perfectly skill-
disjoint but it's reproducible and the per-bucket counts should be
in the 30-80 range (not too lopsided).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from problems import CHALLENGES  # noqa: E402


_SIG_RE = re.compile(r"^def\s+\w+\(.*?\)(?:\s*->\s*.+)?:", re.MULTILINE | re.DOTALL)


def signature(prompt):
    m = _SIG_RE.search(prompt)
    return m.group(0) if m else prompt.split("\n", 1)[0]


# Token-set heuristics based on signature + full prompt body
STRING_HINTS = ("str", "string", "char", "letter", "word", "vowel",
                "consonant", "alphabet", "uppercase", "lowercase",
                "ascii", "palindrome", "encode", "decode")
COLLECTION_HINTS = ("list[", "list ", "list(", ": list", "dict",
                    "set[", "tuple", "array", "sequence", "elements",
                    "the list", "a list", "an array")
NUMBER_HINTS = ("number", "integer", "int ", ": int", "float", "digit",
                "prime", "fibonacci", "factorial", "sum of", "product",
                "divisor", "modulo")


def classify(prompt):
    text = prompt.lower()
    sig = signature(prompt).lower()
    # Priority order: STRINGS > COLLECTIONS > NUMBERS. Strings wins
    # because string-as-element problems often involve a list[str] but
    # the dominant skill is string manipulation.
    if any(h in sig for h in STRING_HINTS) or "string" in text or "char" in text:
        # …but only if the body actually talks about strings, not just
        # using `str()` as a coercion. Require any STRING_HINTS hit in
        # the docstring proper.
        if any(h in text for h in STRING_HINTS):
            return "strings"
    if any(h in sig for h in COLLECTION_HINTS) or any(h in text for h in COLLECTION_HINTS):
        return "collections"
    return "numbers"


def split_subsets():
    subsets = {"strings": [], "numbers": [], "collections": []}
    for ch in CHALLENGES:
        bucket = classify(ch["prompt"])
        subsets[bucket].append(ch)
    return subsets


SUBSETS = split_subsets()


if __name__ == "__main__":
    print(f"\nPhase 15 S2 routing — {len(CHALLENGES)} HumanEval problems\n")
    for name, items in SUBSETS.items():
        print(f"  {name:12s}: {len(items):3d} problems")
        for ch in items[:5]:
            sig = signature(ch["prompt"]).split("\n", 1)[0][:80]
            print(f"    {ch['name']:8s}  {sig}")
        if len(items) > 5:
            print(f"    ... +{len(items)-5} more")
        print()
