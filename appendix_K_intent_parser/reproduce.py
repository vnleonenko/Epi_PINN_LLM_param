#!/usr/bin/env python3
"""Recompute Table A19 and the post hoc check from the archived raw outputs.

No model is called. The script runs

  analyze_intent.py --labels blind_labels_AK.csv blind_labels_VN.csv --labeler-names AK VN
  posthoc_primary_metric.py (same labels)

on results/raw/intent_results.csv, writes to results_rerun/analysis/ and
results_rerun/posthoc/ ($INTENT_OUT/... if INTENT_OUT is set), and compares
every regenerated file with the archived copy in results/. Exit code 0 means
all files are byte-identical.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_intent import rerun_root  # noqa: E402

LABEL_ARGS = ["--labels", "blind_labels_AK.csv", "blind_labels_VN.csv", "--labeler-names", "AK", "VN"]


def compare(archived: Path, regenerated: Path) -> list[str]:
    bad = []
    for ref in sorted(archived.iterdir()):
        new = regenerated / ref.name
        same = new.is_file() and filecmp.cmp(ref, new, shallow=False)
        print(f"  {'same' if same else 'DIFF'}  {ref.relative_to(HERE)}")
        if not same:
            bad.append(str(ref.relative_to(HERE)))
    return bad


def main() -> int:
    out = rerun_root()
    analysis = out / "analysis"
    posthoc = out / "posthoc"
    py = sys.executable
    subprocess.run([py, str(HERE / "analyze_intent.py"), "--out-dir", str(analysis), *LABEL_ARGS], check=True)
    subprocess.run([py, str(HERE / "posthoc_primary_metric.py"), "--out-dir", str(posthoc), *LABEL_ARGS], check=True)
    print("\nComparison with the archived results:")
    bad = compare(HERE / "results" / "analysis", analysis)
    bad += compare(HERE / "results" / "posthoc", posthoc)
    if bad:
        print(f"\nFAILED: {len(bad)} file(s) differ from the archive.")
        return 1
    print("\nOK: all regenerated files are identical to the archived results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
