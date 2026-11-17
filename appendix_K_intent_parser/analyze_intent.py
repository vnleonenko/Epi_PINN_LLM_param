#!/usr/bin/env python3
"""Score an Intent Parser run against gold, the keyword rule and the template rule.

Reads intent_results.csv (and the frozen corpus / baseline), writes:

  summary_table.csv   per-level accuracy + Wilson 95% CIs (LLM, keyword, template)
  summary.md          paper-ready table and test results
  confusion_matrix.csv  7x7 (+ other) for group=single, majority vote
  errors.csv          every majority-incorrect item, with raw outputs
  label_agreement.csv, summary_table_all_items.csv
                      only with --labels: per-item agreement and unfiltered tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Folder layout (all paths are relative to this script, so the folder works
# from the repository root, from inside the folder, or as a standalone copy).
DATA_DIR = HERE / "data"
LABELS_DIR = HERE / "labels"
ARCHIVE_DIR = HERE / "results"            # archived run, never written to
ARCHIVED_RESULTS = ARCHIVE_DIR / "raw" / "intent_results.csv"


def rerun_root() -> Path:
    """Output root for reruns: $INTENT_OUT, else <folder>/results_rerun."""
    env = os.environ.get("INTENT_OUT")
    return Path(env).expanduser().resolve() if env else HERE / "results_rerun"


def resolve_label_path(path: Path) -> Path:
    """Accept a label file given as a path, or just by name from labels/."""
    if path.is_file():
        return path
    candidate = LABELS_DIR / path.name
    return candidate if candidate.is_file() else path


def refuse_archive(out_dir: Path) -> None:
    out = out_dir.expanduser().resolve()
    arch = ARCHIVE_DIR.resolve()
    if out == arch or arch in out.parents:
        raise SystemExit(
            f"Refusing to write into the archived results folder {arch}. "
            "Use --out-dir or INTENT_OUT."
        )

from intent_mapping import (
    LEVEL_ORDER,
    MATRIX_LABELS,
    PARSE_ERROR,
    classify_error,
    exact_mcnemar,
    format_codes,
    is_correct,
    is_exact_match,
    is_lenient_match,
    majority_vote,
    matrix_bucket,
    parse_codes,
    wilson_interval,
)

MATRIX_COLS = MATRIX_LABELS + ["other"]


def default_corpus_path() -> Path:
    return DATA_DIR / "intent_corpus.csv"


def default_baseline_path() -> Path:
    return DATA_DIR / "baseline_results.csv"


def default_results_path(out_dir: Path) -> Path:
    """intent_results.csv in out_dir if present, else the archived run."""
    candidate = out_dir / "intent_results.csv"
    if candidate.is_file():
        return candidate
    return ARCHIVED_RESULTS


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [((n or "").lstrip("\ufeff").strip()) for n in reader.fieldnames]
        return list(reader)


def pct(k: int, n: int) -> str:
    if n == 0:
        return "—"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def fmt_ci(lo: float, hi: float) -> str:
    if any(math.isnan(x) for x in (lo, hi)):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def load_run_meta(out_dir: Path) -> dict:
    path = out_dir / "run_meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def aggregate_items(result_rows: list[dict], corpus: list[dict], baseline: list[dict]) -> list[dict]:
    by_id_corpus = {r["id"]: r for r in corpus}
    by_id_base = {r["id"]: r for r in baseline}
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in result_rows:
        by_id[row["id"]].append(row)

    items = []
    missing_repeats = 0
    for item_id, repeats in sorted(by_id.items()):
        repeats = sorted(repeats, key=lambda r: int(r.get("repeat") or 0))
        corp = by_id_corpus.get(item_id, {})
        base = by_id_base.get(item_id, {})
        gold = parse_codes(corp.get("gold") or (repeats[0].get("gold") if repeats else "none"))
        group = corp.get("group") or repeats[0].get("group", "")
        level = corp.get("level") or repeats[0].get("level", "")
        sets: list[frozenset] = []
        parse_flags: list[bool] = []
        for r in repeats:
            raw_codes = (r.get("parsed_codes") or "").strip()
            failed = (r.get("error_type") == "parse_error") or not raw_codes
            if failed and raw_codes and raw_codes != PARSE_ERROR:
                # parsed_codes is set but error_type says parse_error: error_type wins.
                failed = r.get("error_type") == "parse_error"
            if failed:
                sets.append(frozenset({PARSE_ERROR}))
                parse_flags.append(True)
            else:
                sets.append(parse_codes(raw_codes))
                parse_flags.append(False)

        maj, maj_status = majority_vote(sets)
        if maj is None:
            pred = frozenset({PARSE_ERROR})
            parse_failed = True
            primary, flags = "no_majority", ["no_majority"]
        else:
            parse_failed = PARSE_ERROR in maj
            pred = maj
            primary, flags = classify_error(
                pred, gold, group, level, parse_failed=parse_failed, out_of_set_none_ok=True
            )

        kw = parse_codes(base.get("keyword_rule", "none")) if base else frozenset({"none"})
        tmpl = parse_codes(base.get("template_rule", "none")) if base else frozenset({"none"})

        item = {
            "id": item_id,
            "group": group,
            "level": level,
            "gold": gold,
            "gold_s": format_codes(gold),
            "pred": pred,
            "pred_s": "no_majority" if maj_status == "no_majority" else format_codes(pred),
            "maj_status": maj_status,
            "n_repeats": len(repeats),
            "repeat_sets": [format_codes(s) if PARSE_ERROR not in s else "parse_error" for s in sets],
            "unanimous": maj_status == "unanimous",
            "parse_failed": parse_failed,
            "error_type": primary,
            "error_flags": flags,
            "exact_strict": is_exact_match(pred, gold) and not parse_failed and maj_status != "no_majority",
            "exact": is_correct(pred, gold, group, level, out_of_set_none_ok=True)
            and maj_status != "no_majority",
            "lenient": is_lenient_match(pred, gold, group, level) and maj_status != "no_majority",
            "keyword": kw,
            "keyword_s": format_codes(kw),
            "keyword_ok": is_correct(kw, gold, group, level, out_of_set_none_ok=True),
            "template": tmpl,
            "template_s": format_codes(tmpl),
            "template_ok": is_correct(tmpl, gold, group, level, out_of_set_none_ok=True),
            "raw_outputs": [r.get("raw_output", "") for r in repeats],
            "latencies": [r.get("latency_s", "") for r in repeats],
            "rows": repeats,
        }
        if len(repeats) < 3:
            missing_repeats += 1
        items.append(item)
    if missing_repeats:
        print(
            f"WARNING: {missing_repeats} items have fewer than 3 repeats in the results file.",
            file=sys.stderr,
        )
    # Items that are in the corpus but missing from results
    missing = [r["id"] for r in corpus if r["id"] not in by_id]
    if missing:
        print(
            f"WARNING: {len(missing)} corpus ids have no rows in results "
            f"(e.g. {missing[:5]}). They are omitted from the analysis.",
            file=sys.stderr,
        )
    return items


def acc_block(flags: list[bool]) -> dict:
    k = sum(1 for x in flags if x)
    n = len(flags)
    p, lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi, "label": pct(k, n), "ci": fmt_ci(lo, hi)}


def per_level(items: list[dict]) -> list[dict]:
    by_level: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_level[it["level"] or "?"].append(it)
    levels = [lv for lv in LEVEL_ORDER if lv in by_level] + sorted(
        lv for lv in by_level if lv not in LEVEL_ORDER
    )
    rows = []
    for lv in levels:
        group = by_level[lv]
        llm = acc_block([it["exact"] for it in group])
        llm_strict = acc_block([it["exact_strict"] for it in group])
        llm_lenient = acc_block([it["lenient"] for it in group])
        kw = acc_block([it["keyword_ok"] for it in group])
        tmpl = acc_block([it["template_ok"] for it in group])
        rows.append(
            {
                "level": lv,
                "n": len(group),
                "llm_k": llm["k"],
                "llm_n": llm["n"],
                "llm_acc": llm["p"],
                "llm_ci_lo": llm["lo"],
                "llm_ci_hi": llm["hi"],
                "llm_label": llm["label"],
                "llm_ci": llm["ci"],
                "llm_strict_k": llm_strict["k"],
                "llm_strict_acc": llm_strict["p"],
                "llm_strict_ci": llm_strict["ci"],
                "llm_lenient_k": llm_lenient["k"],
                "llm_lenient_acc": llm_lenient["p"],
                "llm_lenient_ci": llm_lenient["ci"],
                "keyword_k": kw["k"],
                "keyword_n": kw["n"],
                "keyword_acc": kw["p"],
                "keyword_ci_lo": kw["lo"],
                "keyword_ci_hi": kw["hi"],
                "keyword_label": kw["label"],
                "keyword_ci": kw["ci"],
                "template_k": tmpl["k"],
                "template_n": tmpl["n"],
                "template_acc": tmpl["p"],
                "template_ci_lo": tmpl["lo"],
                "template_ci_hi": tmpl["hi"],
                "template_label": tmpl["label"],
                "template_ci": tmpl["ci"],
            }
        )
    return rows


def mcnemar_for(items: list[dict], label: str) -> dict:
    res = exact_mcnemar(
        [it["exact"] for it in items],
        [it["keyword_ok"] for it in items],
    )
    res["subset"] = label
    res["n_items"] = len(items)
    res["llm_k"] = sum(1 for it in items if it["exact"])
    res["keyword_k"] = sum(1 for it in items if it["keyword_ok"])
    return res


def confusion(items: list[dict]) -> list[list]:
    """7x7 + other, majority vote, group=single only."""
    labels = MATRIX_COLS
    index = {name: i for i, name in enumerate(labels)}
    table = [[0 for _ in labels] for _ in labels]
    for it in items:
        if (it["group"] or "").strip().lower() != "single":
            continue
        if it["maj_status"] == "no_majority":
            gold_b = matrix_bucket(it["gold"])
            table[index[gold_b]][index["other"]] += 1
            continue
        gold_b = matrix_bucket(it["gold"])
        pred_b = matrix_bucket(it["pred"])
        table[index[gold_b]][index[pred_b]] += 1
    return table


def write_confusion_csv(path: Path, table: list[list]) -> None:
    labels = MATRIX_COLS
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gold \\ pred"] + labels)
        for i, name in enumerate(labels):
            w.writerow([name] + table[i])


def write_summary_table_csv(path: Path, rows: list[dict], overall: dict, calls: dict) -> None:
    fields = [
        "level",
        "n",
        "llm_k",
        "llm_n",
        "llm_acc",
        "llm_ci_lo",
        "llm_ci_hi",
        "llm_lenient_k",
        "llm_lenient_acc",
        "keyword_k",
        "keyword_n",
        "keyword_acc",
        "keyword_ci_lo",
        "keyword_ci_hi",
        "template_k",
        "template_n",
        "template_acc",
        "template_ci_lo",
        "template_ci_hi",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
        # overall + all-calls rows
        overall_row = {
            "level": "ALL_ITEMS (majority)",
            "n": overall["n"],
            "llm_k": overall["llm"]["k"],
            "llm_n": overall["llm"]["n"],
            "llm_acc": overall["llm"]["p"],
            "llm_ci_lo": overall["llm"]["lo"],
            "llm_ci_hi": overall["llm"]["hi"],
            "llm_lenient_k": overall["lenient"]["k"],
            "llm_lenient_acc": overall["lenient"]["p"],
            "keyword_k": overall["keyword"]["k"],
            "keyword_n": overall["keyword"]["n"],
            "keyword_acc": overall["keyword"]["p"],
            "keyword_ci_lo": overall["keyword"]["lo"],
            "keyword_ci_hi": overall["keyword"]["hi"],
            "template_k": overall["template"]["k"],
            "template_n": overall["template"]["n"],
            "template_acc": overall["template"]["p"],
            "template_ci_lo": overall["template"]["lo"],
            "template_ci_hi": overall["template"]["hi"],
        }
        w.writerow(overall_row)
        w.writerow(
            {
                "level": "ALL_CALLS (333 repeats, no vote)",
                "n": calls["n"],
                "llm_k": calls["k"],
                "llm_n": calls["n"],
                "llm_acc": calls["p"],
                "llm_ci_lo": calls["lo"],
                "llm_ci_hi": calls["hi"],
            }
        )


def md_table(level_rows: list[dict], overall: dict, calls: dict) -> str:
    lines = [
        "| Level | n | LLM (majority) | LLM 95% CI (Wilson) | Keyword rule | Keyword 95% CI | Template rule |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for row in level_rows:
        lines.append(
            f"| {row['level']} | {row['n']} | {row['llm_label']} | {row['llm_ci']} | "
            f"{row['keyword_label']} | {row['keyword_ci']} | {row['template_label']} |"
        )
    o = overall
    lines.append(
        f"| **All items** | {o['n']} | **{o['llm']['label']}** | {o['llm']['ci']} | "
        f"{o['keyword']['label']} | {o['keyword']['ci']} | {o['template']['label']} |"
    )
    lines.append(
        f"| All calls (no vote) | {calls['n']} | {calls['label']} | {calls['ci']} | — | — | — |"
    )
    return "\n".join(lines)


def md_confusion(table: list[list]) -> str:
    labels = MATRIX_COLS
    header = "| gold \\ pred | " + " | ".join(labels) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(labels)) + " |"
    lines = [header, sep]
    for i, name in enumerate(labels):
        lines.append("| " + name + " | " + " | ".join(str(x) for x in table[i]) + " |")
    return "\n".join(lines)


def write_errors_csv(path: Path, items: list[dict]) -> int:
    fields = [
        "id",
        "group",
        "level",
        "gold",
        "pred_majority",
        "maj_status",
        "error_type",
        "error_flags",
        "repeat_codes",
        "keyword_rule",
        "raw_repeat_1",
        "raw_repeat_2",
        "raw_repeat_3",
    ]
    errors = [it for it in items if not it["exact"] or it["maj_status"] == "no_majority"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in errors:
            raws = it["raw_outputs"] + [""] * 3
            w.writerow(
                {
                    "id": it["id"],
                    "group": it["group"],
                    "level": it["level"],
                    "gold": it["gold_s"],
                    "pred_majority": it["pred_s"],
                    "maj_status": it["maj_status"],
                    "error_type": it["error_type"] or "",
                    "error_flags": ";".join(it["error_flags"]),
                    "repeat_codes": " | ".join(it["repeat_sets"]),
                    "keyword_rule": it["keyword_s"],
                    "raw_repeat_1": raws[0],
                    "raw_repeat_2": raws[1],
                    "raw_repeat_3": raws[2],
                }
            )
    return len(errors)



# ---------------------------------------------------------------------------
# Blind labels
# ---------------------------------------------------------------------------

LABEL_COL_CANDIDATES = ("my_label", "label", "gold", "vn_label", "ak_label")
EMPTY_MARKERS = {"", "na", "n/a", "nan", "-", "--", ".", "?", "none?"}


def parse_human_label(value) -> Optional[frozenset]:
    """Parse a human-entered code set. Returns None if the cell is empty / not labeled.

    Accepts ';', commas, surrounding spaces and mixed case. 'none' is a valid label.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.strip().lower() in EMPTY_MARKERS:
        return None
    text = text.replace(",", ";")
    parts = []
    for raw in text.split(";"):
        token = "_".join(raw.strip().lower().split())
        if token and token not in EMPTY_MARKERS:
            parts.append(token)
    if not parts:
        return None
    return frozenset(parts)


def detect_label_column(fieldnames: list[str], label_col: Optional[str] = None) -> str:
    names = list(fieldnames or [])
    if label_col:
        if label_col not in names:
            raise SystemExit(
                f"label column {label_col!r} not found; available: {', '.join(names) or '(none)'}"
            )
        return label_col
    lowered = {n.lower(): n for n in names}
    for cand in LABEL_COL_CANDIDATES:
        if cand in names:
            return cand
        if cand in lowered:
            return lowered[cand]
    raise SystemExit(
        "no label column found (looked for my_label, label, gold). "
        f"Columns: {', '.join(names) or '(none)'}. Pass --label-col."
    )


def flatten_labels_arg(raw) -> list[Path]:
    """Flatten --labels with action='append' and nargs='+'."""
    if not raw:
        return []
    out: list[Path] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            out.extend(resolve_label_path(Path(x)) for x in item)
        else:
            out.append(resolve_label_path(Path(item)))
    return out


def load_label_file(path: Path, label_col: Optional[str] = None) -> tuple[dict[str, Optional[frozenset]], str]:
    if not path.is_file():
        raise SystemExit(f"label file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"label file has no header: {path}")
        reader.fieldnames = [((n or "").lstrip("\ufeff").strip()) for n in reader.fieldnames]
        col = detect_label_column(list(reader.fieldnames), label_col)
        if "id" not in reader.fieldnames:
            raise SystemExit(f"label file {path} has no 'id' column")
        rows = {}
        for row in reader:
            item_id = (row.get("id") or "").strip()
            if not item_id:
                continue
            if item_id in rows:
                print(f"WARNING: duplicate id {item_id} in {path}; using the last row.", file=sys.stderr)
            rows[item_id] = parse_human_label(row.get(col))
    return rows, col


def cohens_kappa(labels_a: list[frozenset], labels_b: list[frozenset]) -> float:
    """Cohen's kappa on exact-set categories for two raters."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return float("nan")
    keys_a = [format_codes(x) for x in labels_a]
    keys_b = [format_codes(x) for x in labels_b]
    p_o = sum(1 for a, b in zip(keys_a, keys_b) if a == b) / n
    ca, cb = Counter(keys_a), Counter(keys_b)
    cats = set(ca) | set(cb)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - p_e) < 1e-15:
        return 1.0 if abs(p_o - 1.0) < 1e-15 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def process_labels(
    *,
    label_paths: list[Path],
    labeler_names: Optional[list[str]],
    label_col: Optional[str],
    empty_excludes: bool,
    corpus: list[dict],
) -> dict:
    if labeler_names and len(labeler_names) != len(label_paths):
        raise SystemExit(
            f"--labeler-names has {len(labeler_names)} values, but --labels has {len(label_paths)} files"
        )
    names: list[str] = []
    used: set[str] = set()
    for i, path in enumerate(label_paths):
        name = (labeler_names[i] if labeler_names else path.stem).strip() or path.stem
        if name in used:
            raise SystemExit(f"duplicate labeler name {name!r}")
        used.add(name)
        names.append(name)

    by_corpus = {r["id"]: r for r in corpus}
    loaded = []
    all_ids: list[str] = []
    seen_ids: set[str] = set()
    for name, path in zip(names, label_paths):
        rows, col = load_label_file(path, label_col)
        loaded.append({"name": name, "path": path, "col": col, "rows": rows})
        for item_id in rows:
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                all_ids.append(item_id)
        unknown = [i for i in rows if i not in by_corpus]
        if unknown:
            print(
                f"WARNING: {name}: {len(unknown)} id(s) not in the corpus "
                f"(e.g. {unknown[:5]}); they are ignored for exclusion.",
                file=sys.stderr,
            )

    # Stable order: corpus order, then any extras
    corpus_ids = [r["id"] for r in corpus]
    ordered_ids = [i for i in corpus_ids if i in seen_ids] + [i for i in all_ids if i not in by_corpus]

    agreement_rows = []
    excluded_ids: list[str] = []
    exclude_reasons: dict[str, str] = {}
    empty_ids: dict[str, list[str]] = {name: [] for name in names}
    agree_flags: dict[str, list[bool]] = {name: [] for name in names}  # only labeled vs gold

    for item_id in ordered_ids:
        corp = by_corpus.get(item_id, {})
        gold = parse_codes(corp["gold"]) if corp else frozenset({"none"})
        group = corp.get("group", "")
        level = corp.get("level", "")
        row = {
            "id": item_id,
            "level": level,
            "group": group,
            "gold": format_codes(gold) if corp else "",
        }
        reasons = []
        any_disagree = False
        any_empty = False
        in_corpus = item_id in by_corpus
        for lab in loaded:
            name = lab["name"]
            present = item_id in lab["rows"]
            raw_set = lab["rows"].get(item_id) if present else None
            if not present or raw_set is None:
                row[f"{name}_label"] = ""
                row[f"{name}_agree"] = ""
                if present:
                    empty_ids[name].append(item_id)
                    any_empty = True
                    if empty_excludes and in_corpus:
                        reasons.append(f"empty:{name}")
                else:
                    row[f"{name}_agree"] = ""
                continue
            row[f"{name}_label"] = format_codes(raw_set)
            if not in_corpus:
                row[f"{name}_agree"] = ""
                continue
            ok = is_correct(raw_set, gold, group, level, out_of_set_none_ok=True)
            row[f"{name}_agree"] = "1" if ok else "0"
            agree_flags[name].append(ok)
            if not ok:
                any_disagree = True
                reasons.append(f"disagree:{name}")
        excluded = bool(in_corpus and (any_disagree or (empty_excludes and any_empty)))
        row["excluded"] = "1" if excluded else "0"
        row["exclude_reason"] = ";".join(reasons) if excluded else ""
        agreement_rows.append(row)
        if excluded:
            excluded_ids.append(item_id)
            exclude_reasons[item_id] = row["exclude_reason"]

    labelers_stats = []
    for lab in loaded:
        name = lab["name"]
        n_rows = len(lab["rows"])
        n_empty = len(empty_ids[name])
        n_labeled = n_rows - n_empty
        flags = agree_flags[name]
        n_agree = sum(1 for x in flags if x)
        n_disagree = sum(1 for x in flags if not x)
        labelers_stats.append(
            {
                "name": name,
                "path": str(lab["path"]),
                "col": lab["col"],
                "n_rows": n_rows,
                "n_labeled": n_labeled,
                "n_empty": n_empty,
                "n_agree": n_agree,
                "n_disagree": n_disagree,
                "agree": acc_block(flags) if flags else acc_block([]),
            }
        )
        if n_empty and not empty_excludes:
            print(
                f"WARNING: {name}: {n_empty} empty label(s) treated as not labeled "
                f"(not excluded). Examples: {empty_ids[name][:5]}. "
                f"Pass --empty-excludes to drop them.",
                file=sys.stderr,
            )

    inter = None
    if len(loaded) >= 2:
        pairs = []
        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                a, b = loaded[i], loaded[j]
                both_ids = [
                    item_id
                    for item_id in ordered_ids
                    if a["rows"].get(item_id) is not None and b["rows"].get(item_id) is not None
                ]
                labs_a = [a["rows"][i] for i in both_ids]
                labs_b = [b["rows"][i] for i in both_ids]
                n_both = len(both_ids)
                n_match = sum(1 for x, y in zip(labs_a, labs_b) if x == y)
                pairs.append(
                    {
                        "a": a["name"],
                        "b": b["name"],
                        "n_both": n_both,
                        "n_agree": n_match,
                        "p_o": (n_match / n_both) if n_both else float("nan"),
                        "kappa": cohens_kappa(labs_a, labs_b),
                    }
                )
        inter = {"pairs": pairs}

    return {
        "names": names,
        "labelers": labelers_stats,
        "excluded_ids": excluded_ids,
        "exclude_reasons": exclude_reasons,
        "empty_ids": empty_ids,
        "empty_excludes": empty_excludes,
        "agreement_rows": agreement_rows,
        "inter": inter,
        "n_ids": len(ordered_ids),
    }


def write_label_agreement_csv(path: Path, report: dict) -> None:
    names = report["names"]
    fields = ["id", "level", "group", "gold"]
    for name in names:
        fields.extend([f"{name}_label", f"{name}_agree"])
    fields.extend(["excluded", "exclude_reason"])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in report["agreement_rows"]:
            w.writerow(row)


def md_label_section(report: dict, n_kept: int, n_all: int) -> list[str]:
    lines = ["## Blind label agreement", ""]
    lines.append("An item is dropped from the main tables if any labeler disagrees")
    lines.append("with corpus gold (exact set match; for group O, `none` also agrees).")
    if report["empty_excludes"]:
        lines.append("Empty labels also exclude the item (`--empty-excludes`).")
    else:
        lines.append("Empty labels are treated as not labeled: they are counted and")
        lines.append("warned about, but do not exclude the item.")
    lines.append("")
    lines.append("| Labeler | File | Labeled | Empty | Agree with gold | 95% CI (Wilson) |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for lab in report["labelers"]:
        agr = lab["agree"]
        agree_s = agr["label"] if lab["n_labeled"] else "—"
        ci_s = agr["ci"] if lab["n_labeled"] else "—"
        lines.append(
            f"| {lab['name']} | `{Path(lab['path']).name}` | "
            f"{lab['n_labeled']}/{lab['n_rows']} | {lab['n_empty']} | "
            f"{agree_s} | {ci_s} |"
        )
    lines.append("")
    if report.get("inter") and report["inter"].get("pairs"):
        lines.append("Inter-labeler agreement is computed on items both labelers filled,")
        lines.append("by exact set match (Cohen's κ on those set-valued categories).")
        lines.append("")
        lines.append("| Pair | n both labeled | Agree with each other | Cohen's κ |")
        lines.append("| --- | ---: | ---: | ---: |")
        for pair in report["inter"]["pairs"]:
            if pair["n_both"] == 0:
                agree_s, kap_s = "—", "—"
            else:
                agree_s = pct(pair["n_agree"], pair["n_both"])
                kap = pair["kappa"]
                kap_s = "—" if (isinstance(kap, float) and math.isnan(kap)) else f"{kap:.3f}"
            lines.append(
                f"| {pair['a']} vs {pair['b']} | {pair['n_both']} | {agree_s} | {kap_s} |"
            )
        lines.append("")
    n_ex = len(report["excluded_ids"])
    if n_ex == 0:
        lines.append("Excluded from the main analysis: **0** items.")
    else:
        lines.append(
            f"Excluded from the main analysis: **{n_ex}** "
            f"item{'s' if n_ex != 1 else ''} (any labeler disagrees with gold"
            + (", or a label was empty" if report["empty_excludes"] else "")
            + "):"
        )
        lines.append("")
        for item_id in report["excluded_ids"]:
            reason = report["exclude_reasons"].get(item_id, "")
            # pull gold / labels from the agreement row
            rec = next((r for r in report["agreement_rows"] if r["id"] == item_id), {})
            bits = [f"gold={rec.get('gold', '')}"]
            for name in report["names"]:
                lab = rec.get(f"{name}_label", "")
                agr = rec.get(f"{name}_agree", "")
                bits.append(f"{name}={lab or '(empty)'}" + ("" if agr == "" else f" (agree={agr})"))
            extra = f" — {reason}" if reason else ""
            lines.append(f"- `{item_id}` ({rec.get('level', '')}; {', '.join(bits)}){extra}")
        lines.append("")
    lines.append(
        f"Per-level tables, McNemar, the confusion matrix and `errors.csv` below "
        f"are computed on the remaining **{n_kept}** of {n_all} items. "
        f"Unfiltered tables: `summary_table_all_items.csv`. "
        f"Item-level comparison: `label_agreement.csv`."
    )
    lines.append("")
    return lines


def compute_item_metrics(items: list[dict]) -> dict:
    level_rows = per_level(items)
    overall = {
        "n": len(items),
        "llm": acc_block([it["exact"] for it in items]),
        "lenient": acc_block([it["lenient"] for it in items]),
        "keyword": acc_block([it["keyword_ok"] for it in items]),
        "template": acc_block([it["template_ok"] for it in items]),
    }
    call_ok = []
    for it in items:
        gold, group, level = it["gold"], it["group"], it["level"]
        for r in it["rows"]:
            raw_codes = (r.get("parsed_codes") or "").strip()
            failed = r.get("error_type") == "parse_error" or not raw_codes
            if failed:
                call_ok.append(False)
            else:
                call_ok.append(
                    is_correct(parse_codes(raw_codes), gold, group, level, out_of_set_none_ok=True)
                )
    by_level = defaultdict(list)
    for it in items:
        by_level[it["level"]].append(it)
    mcnemars = []
    for label, subset in [
        ("L5", by_level.get("L5", [])),
        ("C", by_level.get("C", [])),
        ("overall", items),
    ]:
        if subset:
            mcnemars.append(mcnemar_for(subset, label))
    error_items = [it for it in items if not it["exact"]]
    return {
        "level_rows": level_rows,
        "overall": overall,
        "calls": acc_block(call_ok),
        "consistency": acc_block([it["unanimous"] for it in items]),
        "error_counts": Counter(it["error_type"] or "other" for it in error_items),
        "mcnemars": mcnemars,
        "table": confusion(items),
        "n_items": len(items),
    }


def write_summary_md(
    path: Path,
    *,
    meta: dict,
    level_rows: list[dict],
    overall: dict,
    calls: dict,
    consistency: dict,
    error_counts: Counter,
    mcnemars: list[dict],
    table: list[list],
    n_errors: int,
    n_items: int,
    label_report: Optional[dict] = None,
    n_items_all: Optional[int] = None,
) -> None:
    prompt_src = meta.get("prompt_source", "unknown")
    prompt_hash = meta.get("prompt_sha256", "unknown")
    model = meta.get("model", "unknown")
    ts = meta.get("ended") or meta.get("timestamp") or ""

    lines = []
    lines.append("# Intent Parser pilot test — results")
    lines.append("")
    lines.append("Exact set match of the majority vote (3 repeats) against gold,")
    lines.append("with a Wilson 95% confidence interval per level. For group O")
    lines.append("(out of set), a prediction of `none` (unsupported) is also")
    lines.append("counted correct; the opposite direction is a critical error.")
    lines.append("A lenient metric, gold ⊆ predicted, is reported alongside.")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Prompt source: `{prompt_src}`")
    lines.append(f"- Prompt SHA-256: `{prompt_hash}`")
    if ts:
        lines.append(f"- Timestamp: {ts}")
    if meta.get("temperature") is not None:
        lines.append(f"- Temperature: {meta.get('temperature')}")
    if meta.get("seed") is not None:
        lines.append(f"- Seed: {meta.get('seed')}")
    if meta.get("mock"):
        lines.append(f"- **Mock run** (`--mock {meta.get('mock')}`). Not a real model result.")
    if label_report is not None:
        n_all = n_items_all if n_items_all is not None else n_items
        n_ex = len(label_report.get("excluded_ids") or [])
        lines.append(
            f"- Blind labels: {n_ex} item(s) excluded; tables below use {n_items} of {n_all}."
        )
    lines.append("")
    if label_report is not None:
        lines.extend(
            md_label_section(
                label_report,
                n_kept=n_items,
                n_all=n_items_all if n_items_all is not None else n_items,
            )
        )
    lines.append("## Accuracy by level")
    lines.append("")
    lines.append(md_table(level_rows, overall, calls))
    lines.append("")
    lines.append("Lenient metric (gold ⊆ predicted), all items: ")
    lines.append(f"{overall['lenient']['label']}, Wilson {overall['lenient']['ci']}.")
    lines.append("")
    lines.append("## Repeat consistency")
    lines.append("")
    lines.append(
        f"{consistency['k']}/{consistency['n']} items "
        f"({100.0 * consistency['k'] / consistency['n']:.1f}%) "
        "had all repeats identical (including unanimous parse errors)."
    )
    lines.append("")
    lines.append("## Error types (majority vote, incorrect items)")
    lines.append("")
    if n_errors == 0:
        lines.append("No incorrect items.")
    else:
        lines.append("| Type | Count |")
        lines.append("| --- | ---: |")
        for name, count in error_counts.most_common():
            note = ""
            if name == "opposite_direction":
                note = " (critical)"
            lines.append(f"| {name}{note} | {count} |")
        lines.append("")
        lines.append(f"Total incorrect items: {n_errors} / {n_items}. Full list: `errors.csv`.")
    lines.append("")
    lines.append("## Exact McNemar test (LLM vs keyword rule)")
    lines.append("")
    lines.append("Two-sided exact test on discordant pairs (binomial, p = 1/2).")
    lines.append("`b` = LLM correct and keyword wrong; `c` = the reverse.")
    lines.append("")
    lines.append("| Subset | n | LLM | Keyword | b | c | p |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in mcnemars:
        lines.append(
            f"| {m['subset']} | {m['n_items']} | {m['llm_k']}/{m['n_items']} | "
            f"{m['keyword_k']}/{m['n_items']} | {m['b_llm_only']} | "
            f"{m['c_baseline_only']} | {fmt_p(m['p_value'])} |"
        )
    lines.append("")
    lines.append("## Confusion matrix (group = single, majority vote)")
    lines.append("")
    lines.append("Rows are gold, columns are predicted. The six paper targets plus")
    lines.append("`none`. Multi-label or out-of-set predictions are binned as `other`.")
    lines.append("")
    lines.append(md_confusion(table))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Gold codes in the corpus are separated by a single `;`.")
    lines.append("- `primary_metric` and `reasoning` are ignored by the mapping.")
    lines.append("- Direction `any` on a cared-about axis contributes no code.")
    lines.append("- The `correct` column in `intent_results.csv` is a strict exact")
    lines.append("  set match per repeat (group-O special rule is applied only here).")
    lines.append("- Template rule and keyword rule are the frozen baselines in")
    lines.append("  `keyword_baseline.py` / `baseline_results.csv`.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> int:
    out_dir = (args.out_dir or (rerun_root() / "analysis")).expanduser().resolve()
    refuse_archive(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results or default_results_path(out_dir)
    if not results_path.is_file():
        raise SystemExit(f"results file not found: {results_path}")
    if not args.corpus.is_file():
        raise SystemExit(f"corpus not found: {args.corpus}")
    if not args.baseline.is_file():
        print(f"WARNING: baseline CSV not found at {args.baseline}; keyword/template will be empty/none.", file=sys.stderr)

    results = load_csv(results_path)
    corpus = load_csv(args.corpus)
    baseline = load_csv(args.baseline) if args.baseline.is_file() else []
    if not results:
        raise SystemExit(f"no rows in {results_path}")

    all_items = aggregate_items(results, corpus, baseline)
    if not all_items:
        raise SystemExit("no items to analyse")

    label_paths = flatten_labels_arg(getattr(args, "labels", None))
    label_report = None
    items = all_items
    if label_paths:
        label_report = process_labels(
            label_paths=label_paths,
            labeler_names=getattr(args, "labeler_names", None),
            label_col=getattr(args, "label_col", None),
            empty_excludes=bool(getattr(args, "empty_excludes", False)),
            corpus=corpus,
        )
        excluded = set(label_report["excluded_ids"])
        items = [it for it in all_items if it["id"] not in excluded]
        write_label_agreement_csv(out_dir / "label_agreement.csv", label_report)
        metrics_all = compute_item_metrics(all_items)
        write_summary_table_csv(
            out_dir / "summary_table_all_items.csv",
            metrics_all["level_rows"],
            metrics_all["overall"],
            metrics_all["calls"],
        )
        print(
            f"labels: excluded {len(excluded)} item(s); "
            f"main tables use {len(items)} of {len(all_items)}"
        )
        if excluded:
            print("  excluded ids: " + ", ".join(label_report["excluded_ids"]))

    metrics = compute_item_metrics(items)
    level_rows = metrics["level_rows"]
    overall = metrics["overall"]
    calls = metrics["calls"]
    consistency = metrics["consistency"]
    error_counts = metrics["error_counts"]
    mcnemars = metrics["mcnemars"]
    table = metrics["table"]

    meta = load_run_meta(out_dir)
    if not meta:
        # run_meta.json usually sits next to the results file.
        meta = load_run_meta(results_path.parent)

    write_summary_table_csv(out_dir / "summary_table.csv", level_rows, overall, calls)
    write_confusion_csv(out_dir / "confusion_matrix.csv", table)
    n_errors = write_errors_csv(out_dir / "errors.csv", items)
    write_summary_md(
        out_dir / "summary.md",
        meta=meta,
        level_rows=level_rows,
        overall=overall,
        calls=calls,
        consistency=consistency,
        error_counts=error_counts,
        mcnemars=mcnemars,
        table=table,
        n_errors=n_errors,
        n_items=len(items),
        label_report=label_report,
        n_items_all=len(all_items),
    )

    print(f"items: {len(items)}  calls in file: {len(results)}"
          + (f"  (unfiltered {len(all_items)})" if label_report is not None else ""))
    print(f"LLM majority exact: {overall['llm']['label']}  CI {overall['llm']['ci']}")
    print(f"Keyword:            {overall['keyword']['label']}  CI {overall['keyword']['ci']}")
    print(f"Template:           {overall['template']['label']}  CI {overall['template']['ci']}")
    print(f"Lenient (gold ⊆ pred): {overall['lenient']['label']}")
    print(f"All calls: {calls['label']}  CI {calls['ci']}")
    print(f"Consistency 3/3: {consistency['label']}")
    for m in mcnemars:
        print(
            f"McNemar {m['subset']}: b={m['b_llm_only']} c={m['c_baseline_only']} p={m['p_value']:.4g}"
        )
    print(f"wrote {out_dir / 'summary_table.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"wrote {out_dir / 'confusion_matrix.csv'}")
    print(f"wrote {out_dir / 'errors.csv'} ({n_errors} incorrect items)")
    if label_report is not None:
        print(f"wrote {out_dir / 'label_agreement.csv'}")
        print(f"wrote {out_dir / 'summary_table_all_items.csv'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=None, help="intent_results.csv. Default: <out-dir>/intent_results.csv if present, else the archived results/raw/intent_results.csv")
    p.add_argument("--corpus", type=Path, default=default_corpus_path())
    p.add_argument("--baseline", type=Path, default=default_baseline_path())
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write summary_table.csv, summary.md, confusion_matrix.csv, errors.csv. Default: results_rerun/analysis/ in this folder ($INTENT_OUT/analysis/ if INTENT_OUT is set). The archived results/ folder is never written to.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        action="append",
        type=Path,
        default=None,
        help="Blind-label CSV(s) with at least columns id and my_label (or --label-col); a bare file name is looked up in labels/. Repeatable. An item is dropped from the main tables if any labeler disagrees with gold.",
    )
    p.add_argument(
        "--label-col",
        default=None,
        help="Label column name. Default: my_label, else label, else gold.",
    )
    p.add_argument(
        "--labeler-names",
        nargs="+",
        default=None,
        help="Names for the --labels files, in the same order. Default: file stems.",
    )
    p.add_argument(
        "--empty-excludes",
        action="store_true",
        help="Exclude items a labeler left empty. Default: warn only, do not exclude.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.out_dir is None:
        args.out_dir = rerun_root() / "analysis"
    if args.results is None:
        args.results = default_results_path(args.out_dir)
    return analyze(args)


if __name__ == "__main__":
    sys.exit(main())
