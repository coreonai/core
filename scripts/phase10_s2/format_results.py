"""Phase 10 S2: format scripts/phase10_s2/results.tsv into a markdown
table block, suitable to paste into the README.

Usage: python format_results.py [path/to/results.tsv]"""

import csv
import sys
from pathlib import Path


def main():
    tsv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/phase10_s2/results.tsv")
    if not tsv_path.exists():
        print(f"missing {tsv_path}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    if not rows:
        print("results.tsv is empty", file=sys.stderr)
        sys.exit(0)

    header = "| variant | pass | mean-AUC | sum-AUC | F=2 lift | F=4 lift | F=8 lift | F=16 lift |"
    sep    = "|---------|-----:|---------:|--------:|---------:|---------:|---------:|----------:|"
    print(header)
    print(sep)
    for r in rows:
        def fmt(key, decimals=3):
            v = r.get(key, "")
            try:
                return f"{float(v):.{decimals}f}"
            except (ValueError, TypeError):
                return v or "—"
        def fmt_lift(key):
            v = r.get(key, "")
            try:
                return f"{float(v):.2f}×"
            except (ValueError, TypeError):
                return v or "—"
        print(
            f"| {r['name']:<20} | {fmt('pass_rate', 3)} | {fmt('mean_auc', 3)} | "
            f"{fmt('sum_auc', 3)} | {fmt_lift('F2_lift')} | {fmt_lift('F4_lift')} | "
            f"{fmt_lift('F8_lift')} | {fmt_lift('F16_lift')} |"
        )


if __name__ == "__main__":
    main()
