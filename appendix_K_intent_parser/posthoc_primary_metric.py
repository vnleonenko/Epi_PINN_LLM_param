#!/usr/bin/env python3
"""Post hoc sensitivity analysis of the primary_metric parse failures.

This check is not part of the protocol fixed before the run. After the run,
every parse_error turned out to come from one Pydantic check:
ExpertIntent.primary_metric allows only position/height/both/growth/mortality,
so when the model answers "no change" with primary_metric "none" or "any",
validation fails even though all cares_* / *_direction fields are valid.

This script re-reads the saved raw outputs (no model calls), and for rows whose
only validation error is the primary_metric literal, maps the remaining fields
with the frozen mapping (intent_mapping.map_intent_to_codes; primary_metric is
ignored by that mapping). All other rows are copied unchanged.

Input: the archived raw outputs results/raw/intent_results.csv (or --results).
Output (separate folder; the archived results/ folder is never written to):
  results_rerun/posthoc/intent_results_posthoc.csv
  results_rerun/posthoc/recovered_rows.csv
  results_rerun/posthoc/summary.md ... (via analyze_intent.py)
Set INTENT_OUT to write under $INTENT_OUT/posthoc instead, or use --out-dir.

Usage (from the repository root):
  python appendix_K_intent_parser/posthoc_primary_metric.py --labels blind_labels_AK.csv blind_labels_VN.csv --labeler-names AK VN
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_intent import ARCHIVED_RESULTS, refuse_archive, rerun_root  # noqa: E402
from intent_mapping import PARSE_ERROR, classify_error, format_codes, is_exact_match, map_intent_to_codes, parse_codes  # noqa: E402

PM_ERR = re.compile(r"validation error.*primary_metric.*literal_error", re.S)
ONE_ERR = re.compile(r"^parse: 1 validation error", re.S)


def extract_json(raw: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    text = m.group(1) if m else raw[raw.find("{"): raw.rfind("}") + 1]
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ARCHIVED_RESULTS))
    ap.add_argument("--out-dir", default=str(rerun_root() / "posthoc"))
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--labeler-names", nargs="*", default=[])
    args = ap.parse_args()

    out = Path(args.out_dir)
    refuse_archive(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.results, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    recovered = []
    for r in rows:
        err = r.get("mapping_error") or ""
        if r.get("error_type") != "parse_error" or not (ONE_ERR.search(err) and PM_ERR.search(err)):
            continue
        try:
            data = extract_json(r["raw_output"])
        except Exception as exc:  # leave as parse_error
            print(f"{r['id']} r{r['repeat']}: JSON not recoverable ({exc})", file=sys.stderr)
            continue
        codes, merr = map_intent_to_codes(data)
        if merr or PARSE_ERROR in codes:
            continue
        gold = parse_codes(r.get("gold", ""))
        old = r["parsed_codes"] or "parse_error"
        r["parsed_codes"] = format_codes(codes)
        r["error_type"], _ = classify_error(codes, gold, r.get("group", ""), r.get("level", ""))
        r["correct"] = "1" if is_exact_match(codes, gold) else "0"
        r["mapping_error"] = "POSTHOC: primary_metric=%r ignored; was %s" % (data.get("primary_metric"), old)
        recovered.append({"id": r["id"], "repeat": r["repeat"], "level": r.get("level", ""),
                          "gold": r.get("gold", ""), "primary_metric": data.get("primary_metric"),
                          "new_codes": r["parsed_codes"], "correct_strict": r["correct"]})

    res_path = out / "intent_results_posthoc.csv"
    with open(res_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out / "recovered_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "repeat", "level", "gold", "primary_metric", "new_codes", "correct_strict"])
        w.writeheader()
        w.writerows(recovered)
    print(f"recovered {len(recovered)} row(s) from primary_metric-only parse errors")

    cmd = [sys.executable, str(HERE / "analyze_intent.py"), "--results", str(res_path), "--out-dir", str(out)]
    if args.labels:
        cmd += ["--labels", *args.labels]
    if args.labeler_names:
        cmd += ["--labeler-names", *args.labeler_names]
    rc = subprocess.call(cmd)
    md = out / "summary.md"
    if md.exists():
        md.write_text("> **POST HOC sensitivity analysis, not pre-registered.** Parse errors caused only by "
                      "`primary_metric` = none/any are re-mapped from the saved raw outputs; everything else is "
                      "identical to the main analysis. See `recovered_rows.csv`.\n\n" + md.read_text(encoding="utf-8"),
                      encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
