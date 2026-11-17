#!/usr/bin/env python3
"""L5 without the two requests shown to the labelers (Appendix K, Results).

Q068 and Q077 were shown as examples in the labeling instructions
(Appendix K of the article).
This script drops them, together with the two items excluded by blind
labeling (Q067, Q107), and repeats the L5 comparison of the Intent Parser
with the keyword rule (exact McNemar). Reads the archived raw outputs by
default; no model is called and nothing is written.

Expected output: L5 without Q068/Q077: LLM 5/9, keyword 0/9, b=5 c=0, p=0.0625
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_intent import ARCHIVED_RESULTS, DATA_DIR, LABELS_DIR, aggregate_items, load_csv  # noqa: E402
from intent_mapping import exact_mcnemar  # noqa: E402

SHOWN_AS_EXAMPLES = {"Q068", "Q077"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ARCHIVED_RESULTS)
    args = ap.parse_args()
    items = aggregate_items(
        load_csv(args.results),
        load_csv(DATA_DIR / "intent_corpus.csv"),
        load_csv(DATA_DIR / "baseline_results.csv"),
    )
    with (LABELS_DIR / "excluded_items.csv").open(encoding="utf-8", newline="") as f:
        excluded = {r["id"] for r in csv.DictReader(f) if r.get("excluded") == "1"}
    l5 = [it for it in items if it["level"] == "L5" and it["id"] not in excluded | SHOWN_AS_EXAMPLES]
    llm = [bool(it["exact"]) for it in l5]
    kw = [bool(it["keyword_ok"]) for it in l5]
    res = exact_mcnemar(llm, kw)
    print(
        f"L5 without {', '.join(sorted(SHOWN_AS_EXAMPLES))} (and excluded {', '.join(sorted(excluded))}): "
        f"LLM {sum(llm)}/{len(l5)}, keyword {sum(kw)}/{len(l5)}, "
        f"b={res['b_llm_only']} c={res['c_baseline_only']}, p={res['p_value']:.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
