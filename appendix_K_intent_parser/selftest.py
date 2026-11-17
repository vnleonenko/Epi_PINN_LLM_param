#!/usr/bin/env python3
"""Unit tests and mock end-to-end runs for the Appendix K scripts.

Needs only pydantic; no model is called. Artifacts go to
results_rerun/selftest/ ($INTENT_OUT/selftest if INTENT_OUT is set).
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundled_parser import parse_intent
from analyze_intent import cohens_kappa, parse_human_label
from intent_mapping import (
    MAPPING,
    codes_to_intent_dict,
    exact_mcnemar,
    format_codes,
    is_correct,
    map_intent_to_codes,
    parse_codes,
    wilson_interval,
    classify_error,
    majority_vote,
)

from bundled_parser import FROZEN_PROMPT_SHA256, ExpertIntent, load_system_prompt, prompt_sha256
from run_intent_test import parse_one, build_prompt_bundle, rerun_root

PY = sys.executable
SELFTEST_DIR = rerun_root() / "selftest"
DATA_DIR = HERE / "data"
ARCHIVE_DIR = HERE / "results"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_mapping_roundtrip() -> None:
    singles = list(MAPPING.values())
    compounds = [
        {"peak_higher", "peak_earlier"},
        {"peak_lower", "peak_later"},
        {"growth_faster", "mortality_higher"},
        {"none"},
        set(),
    ]
    for codes in [{c} for c in singles] + compounds:
        intent = codes_to_intent_dict(codes, reasoning="roundtrip")
        back, err = map_intent_to_codes(intent)
        check(err is None, f"mapping error for {codes}: {err}")
        expected = frozenset(codes) if codes and codes != {"none"} and codes != set() else frozenset({"none"})
        if not codes or codes == {"none"}:
            expected = frozenset({"none"})
        check(back == expected, f"roundtrip failed: {codes} -> {back}")
    # cares=true with direction "any" gives no code
    intent = codes_to_intent_dict({"peak_higher"})
    intent["cares_about_position"] = True
    intent["position_direction"] = "any"
    back, err = map_intent_to_codes(intent)
    check(err is None and back == frozenset({"peak_higher"}), f"any leaked: {back} {err}")
    # unmapped direction is parse error
    intent = codes_to_intent_dict({"peak_higher"})
    intent["height_direction"] = "sideways"
    back, err = map_intent_to_codes(intent)
    check(err is not None, "unmapped direction should error")
    # gold separator
    check(parse_codes("peak_higher;peak_earlier") == frozenset({"peak_higher", "peak_earlier"}), "sep")
    check(format_codes({"peak_earlier", "peak_higher"}) == "peak_earlier;peak_higher", "format sort")
    print("OK mapping roundtrip")


def test_wilson() -> None:
    p, lo, hi = wilson_interval(12, 12)
    check(abs(p - 1.0) < 1e-12, "p")
    check(abs(lo - 0.7575) < 0.002, f"12/12 lo={lo} expected ~0.7575")
    check(abs(hi - 1.0) < 1e-9, f"12/12 hi={hi}")
    p0, lo0, hi0 = wilson_interval(0, 12)
    check(p0 == 0.0 and abs(hi0 - 0.2425) < 0.002, f"0/12 hi={hi0}")
    print(f"OK Wilson 12/12 CI=[{lo:.4f}, {hi:.4f}]")


def test_mcnemar() -> None:
    # Five discordant pairs (LLM right, keyword wrong) and three concordant ones:
    # b=5, c=0, so p = 2 * (1/2)**5 = 0.0625.
    llm = [True] * 5 + [True, False, True]
    kw = [False] * 5 + [True, False, True]
    res = exact_mcnemar(llm, kw)
    check(res["b_llm_only"] == 5 and res["c_baseline_only"] == 0, res)
    check(abs(res["p_value"] - 0.0625) < 1e-12, res)
    res0 = exact_mcnemar([True, True], [True, True])
    check(res0["p_value"] == 1.0 and res0["n_discordant"] == 0, res0)
    print("OK McNemar")


def test_error_types() -> None:
    et, flags = classify_error({"peak_lower"}, {"peak_higher"})
    check(et == "opposite_direction", et)
    et, flags = classify_error({"peak_later"}, {"peak_lower"})
    check(et == "wrong_axis", f"{et} {flags}")
    et, flags = classify_error({"peak_lower", "peak_later"}, {"peak_lower"})
    check(et == "extra_target", f"{et} {flags}")
    et, flags = classify_error({"peak_higher"}, {"peak_higher", "peak_earlier"}, group="compound")
    check(et == "missing_target", f"{et} {flags}")
    et, flags = classify_error({"peak_higher"}, {"none"}, group="none")
    check(et == "false_positive", f"{et} {flags}")
    et, flags = classify_error({"none"}, {"none"}, group="none")
    check(et == "", flags)
    et, flags = classify_error({"none"}, {"growth_slower"}, group="out_of_set", level="O")
    check(et == "", f"O none should be correct, got {et} {flags}")
    et, flags = classify_error({"growth_faster"}, {"growth_slower"}, group="out_of_set", level="O")
    check(et == "opposite_direction", f"{et} {flags}")
    check(is_correct({"none"}, {"mortality_lower"}, "out_of_set", "O"), "O none should be correct")
    check(not is_correct({"none"}, {"mortality_lower"}, "out_of_set", "O", out_of_set_none_ok=False), "strict O none should fail")
    maj, status = majority_vote([{"a"}, {"a"}, {"b"}])
    check(maj == frozenset({"a"}) and status == "majority", (maj, status))
    maj, status = majority_vote([{"a"}, {"b"}, {"c"}])
    check(maj is None and status == "no_majority", (maj, status))
    print("OK error types / majority / group O")


def run_cmd(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(HERE), check=True, text=True, capture_output=True, **kw)


def test_mock_e2e() -> None:
    if SELFTEST_DIR.exists():
        shutil.rmtree(SELFTEST_DIR)

    # Resume: limit 5, then complete.
    r = run_cmd(
        [PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(SELFTEST_DIR / "gold_partial"), "--limit", "5"]
    )
    csv_path = SELFTEST_DIR / "gold_partial" / "intent_results.csv"
    with csv_path.open(encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    check(n == 5, f"partial should have 5 rows, got {n}")

    r = run_cmd(
        [PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(SELFTEST_DIR / "gold_partial")]
    )
    check("already done" in (r.stdout + r.stderr) or "328 remaining" in r.stdout or "333 remaining" not in r.stdout, r.stdout)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 333, f"after resume expected 333, got {len(rows)}")
    ids = {(row["id"], int(row["repeat"])) for row in rows}
    check(len(ids) == 333, "duplicate keys after resume")

    # Gold analysis: 111/111
    gold_dir = SELFTEST_DIR / "gold"
    run_cmd([PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(gold_dir), "--no-resume"])
    run_cmd(
        [
            PY,
            "analyze_intent.py",
            "--out-dir",
            str(gold_dir),
            "--results",
            str(gold_dir / "intent_results.csv"),
        ]
    )
    summary = (gold_dir / "summary.md").read_text(encoding="utf-8")
    check("111/111" in summary, "gold mock should be 111/111")
    errors = list(csv.DictReader((gold_dir / "errors.csv").open(encoding="utf-8")))
    check(len(errors) == 0, f"gold mock should have 0 errors, got {len(errors)}")
    meta = json.loads((gold_dir / "run_meta.json").read_text())
    check(meta.get("prompt_sha256"), "missing prompt hash")
    check((gold_dir / "prompt_used.txt").is_file(), "missing prompt_used.txt")
    check("Mock run" in summary, "summary should flag mock")

    # Keyword mock must match frozen baseline
    kw_dir = SELFTEST_DIR / "keyword"
    run_cmd([PY, "run_intent_test.py", "--mock", "keyword", "--out-dir", str(kw_dir), "--no-resume"])
    run_cmd(
        [PY, "analyze_intent.py", "--out-dir", str(kw_dir), "--results", str(kw_dir / "intent_results.csv")]
    )
    table = list(csv.DictReader((kw_dir / "summary_table.csv").open(encoding="utf-8")))
    by_level = {r["level"]: r for r in table}
    expected = {
        "L0": 6, "L1": 18, "L2": 15, "L3": 10, "L4": 12,
        "L5": 0, "RU": 11, "C": 6, "H": 2, "N": 8, "O": 4,
    }
    for lv, k in expected.items():
        got = int(float(by_level[lv]["llm_k"]))
        kwk = int(float(by_level[lv]["keyword_k"]))
        n = int(float(by_level[lv]["n"]))
        check(got == k == kwk, f"keyword mock level {lv}: llm={got} keyword={kwk} expected={k}/{n}")
    # overall 92/111 (6+18+15+10+12+0+11+6+2+8+4 = 92)
    overall = by_level["ALL_ITEMS (majority)"]
    check(int(float(overall["llm_k"])) == 92, f"keyword overall {overall['llm_k']}")
    check(int(float(overall["template_k"])) == 18, f"template overall {overall['template_k']} expected 18")
    check(int(float(by_level["O"]["template_k"])) == 4, "template O should be 4/4 (group-O rule: none = unsupported counts as correct)")
    print("OK keyword mock matches frozen baseline (92/111)")

    # Mixed mock: injected error types
    mx_dir = SELFTEST_DIR / "mixed"
    run_cmd([PY, "run_intent_test.py", "--mock", "mixed", "--out-dir", str(mx_dir), "--no-resume"])
    run_cmd(
        [PY, "analyze_intent.py", "--out-dir", str(mx_dir), "--results", str(mx_dir / "intent_results.csv")]
    )
    errors = {r["id"]: r for r in csv.DictReader((mx_dir / "errors.csv").open(encoding="utf-8"))}
    check(errors["Q002"]["error_type"] == "parse_error", errors["Q002"])
    check(errors["Q013"]["error_type"] == "opposite_direction", errors["Q013"])
    check(errors["Q027"]["error_type"] == "wrong_axis", errors["Q027"])
    check(errors["Q098"]["error_type"] == "extra_target", errors["Q098"])
    check(errors["Q091"]["error_type"] == "missing_target", errors["Q091"])
    check(errors["Q100"]["error_type"] == "false_positive", errors["Q100"])
    check(errors["Q108"]["error_type"] == "opposite_direction", errors["Q108"])
    check("Q109" not in errors, "Q109 none should be correct under O-rule")
    check("Q005" not in errors, "Q005 majority should still be gold")
    # consistency not 111/111 because Q005 has a dissenting repeat
    mx_summary = (mx_dir / "summary.md").read_text(encoding="utf-8")
    check("110/111" in mx_summary or "109/111" in mx_summary, "consistency should mention the dissent")
    print("OK mixed mock error types")

    # Refuse mock writes into the experiment folder
    bad = subprocess.run(
        [PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(HERE)],
        cwd=str(HERE),
        text=True,
        capture_output=True,
    )
    check(bad.returncode != 0, "should refuse mock into experiment dir")
    check("Refusing" in (bad.stdout + bad.stderr), bad.stderr)
    print("OK refuse mock into experiment folder")

    # --print-prompt needs no model
    r = run_cmd([PY, "run_intent_test.py", "--mock", "gold", "--print-prompt"])
    check("You are an epidemiologist" in r.stdout, "prompt missing")
    check("cares_about_growth" in r.stdout, "growth axis missing from bundled prompt")
    print("OK print-prompt")

    # parse_intent accepts fenced JSON
    blob = codes_to_intent_dict({"peak_later"})
    fenced = "```json\n" + json.dumps(blob) + "\n```"
    parsed = parse_intent(fenced)
    back, err = map_intent_to_codes(parsed)
    check(err is None and back == frozenset({"peak_later"}), (back, err))
    print("OK fenced JSON parse")

    # No mock artifacts in the experiment root
    for name in ["intent_results.csv", "summary.md", "summary_table.csv", "errors.csv", "prompt_used.txt", "run_meta.json"]:
        check(not (HERE / name).exists(), f"mock output leaked into {HERE / name}")
    print("OK no leak into experiment folder")


def test_label_helpers() -> None:
    check(parse_human_label("") is None, "empty")
    check(parse_human_label("  ") is None, "blank")
    check(parse_human_label("none") == frozenset({"none"}), "none is a label")
    check(parse_human_label("Peak_Higher") == frozenset({"peak_higher"}), "case")
    check(
        parse_human_label(" peak higher ; peak earlier ")
        == frozenset({"peak_higher", "peak_earlier"}),
        "spaces and semicolon",
    )
    check(
        parse_human_label("peak_lower, peak_later")
        == frozenset({"peak_lower", "peak_later"}),
        "comma",
    )
    # Known example: A=a,a,a,b and B=a,a,b,b give kappa = 0.5
    a = [frozenset({x}) for x in list("aaab")]
    b = [frozenset({x}) for x in list("aabb")]
    kap = cohens_kappa(a, b)
    check(abs(kap - 0.5) < 1e-12, f"kappa={kap} expected 0.5")
    check(abs(cohens_kappa(a, a) - 1.0) < 1e-12, "perfect kappa")
    print("OK label parsing and Cohen's kappa")


def _write_label_csv(path: Path, rows: list[dict], col: str = "my_label") -> None:
    fields = ["id", col, "comment"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def test_label_filter() -> None:
    """Fake AK/VN labels on the gold mock: one disagreement, an O-group none, one empty."""
    gold_dir = SELFTEST_DIR / "gold"
    if not (gold_dir / "intent_results.csv").is_file():
        run_cmd([PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(gold_dir), "--no-resume"])

    corpus = list(csv.DictReader((DATA_DIR / "intent_corpus.csv").open(encoding="utf-8-sig")))
    # AK: gold everywhere except Q013 (wrong), Q109 = none (group O, counts as agreement) and Q067 (empty)
    ak_rows = []
    vn_rows = []
    for row in corpus:
        gold = row["gold"]
        ak_label = gold
        vn_label = gold
        if row["id"] == "Q013":
            ak_label = "peak_lower"  # gold is peak_higher
            vn_label = gold
        if row["id"] == "Q109":
            ak_label = "none"  # O: none agrees with mortality_lower
            vn_label = gold  # exact gold; inter-labeler mismatch with AK
        if row["id"] == "Q067":
            ak_label = ""  # empty: warn, do not exclude
            vn_label = gold
        ak_rows.append({"id": row["id"], "my_label": ak_label})
        vn_rows.append({"id": row["id"], "my_label": vn_label})
    lab_dir = SELFTEST_DIR / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    ak_path = lab_dir / "blind_labels_AK.csv"
    vn_path = lab_dir / "blind_labels_VN.csv"
    _write_label_csv(ak_path, ak_rows)
    _write_label_csv(vn_path, vn_rows)

    out = SELFTEST_DIR / "gold_filtered"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    # copy results/meta so analyze can find them in out-dir
    for name in ("intent_results.csv", "run_meta.json", "prompt_used.txt"):
        src = gold_dir / name
        if src.is_file():
            shutil.copy(src, out / name)

    # default: empty does not exclude; Q013 excluded; Q109 kept
    run_cmd(
        [
            PY,
            "analyze_intent.py",
            "--out-dir",
            str(out),
            "--results",
            str(out / "intent_results.csv"),
            "--labels",
            str(ak_path),
            str(vn_path),
            "--labeler-names",
            "AK",
            "VN",
        ]
    )
    agree = {r["id"]: r for r in csv.DictReader((out / "label_agreement.csv").open(encoding="utf-8"))}
    check(agree["Q013"]["excluded"] == "1", "Q013 must be excluded (AK disagrees)")
    check(agree["Q013"]["AK_agree"] == "0" and agree["Q013"]["VN_agree"] == "1", agree["Q013"])
    check(agree["Q109"]["excluded"] == "0", "Q109 none must agree with gold on group O")
    check(agree["Q109"]["AK_label"] == "none" and agree["Q109"]["AK_agree"] == "1", agree["Q109"])
    check(agree["Q067"]["excluded"] == "0", "empty must not exclude by default")
    check(agree["Q067"]["AK_label"] == "" and agree["Q067"]["AK_agree"] == "", agree["Q067"])

    table = {r["level"]: r for r in csv.DictReader((out / "summary_table.csv").open(encoding="utf-8"))}
    all_tab = {r["level"]: r for r in csv.DictReader((out / "summary_table_all_items.csv").open(encoding="utf-8"))}
    check(int(float(all_tab["ALL_ITEMS (majority)"]["n"])) == 111, "unfiltered n")
    check(int(float(table["ALL_ITEMS (majority)"]["n"])) == 110, f"filtered n={table['ALL_ITEMS (majority)']['n']}")
    # L3 has 12 items, one excluded (Q013)
    check(int(float(table["L3"]["n"])) == 11, f"L3 n={table['L3']['n']}")
    check(int(float(all_tab["L3"]["n"])) == 12, "unfiltered L3")
    # An excluded item must not appear in errors.csv
    err_ids = {r["id"] for r in csv.DictReader((out / "errors.csv").open(encoding="utf-8"))}
    check("Q013" not in err_ids, "excluded id must not appear in errors.csv")
    summary = (out / "summary.md").read_text(encoding="utf-8")
    check("Blind label agreement" in summary, "section missing")
    check("Q013" in summary, "excluded id should be listed")
    check("Cohen" in summary or "κ" in summary, "kappa missing")
    # AK and VN disagree on Q013 and Q109; Q067 is labeled by VN only.

    # --empty-excludes should also drop Q067
    out2 = SELFTEST_DIR / "gold_filtered_empty"
    if out2.exists():
        shutil.rmtree(out2)
    out2.mkdir()
    for name in ("intent_results.csv", "run_meta.json", "prompt_used.txt"):
        src = gold_dir / name
        if src.is_file():
            shutil.copy(src, out2 / name)
    run_cmd(
        [
            PY,
            "analyze_intent.py",
            "--out-dir",
            str(out2),
            "--results",
            str(out2 / "intent_results.csv"),
            "--labels",
            str(ak_path),
            "--labeler-names",
            "AK",
            "--empty-excludes",
        ]
    )
    agree2 = {r["id"]: r for r in csv.DictReader((out2 / "label_agreement.csv").open(encoding="utf-8"))}
    check(agree2["Q067"]["excluded"] == "1", "empty-excludes should drop Q067")
    check(agree2["Q013"]["excluded"] == "1", "Q013 still excluded")
    check(agree2["Q109"]["excluded"] == "0", "Q109 still kept")
    table2 = {r["level"]: r for r in csv.DictReader((out2 / "summary_table.csv").open(encoding="utf-8"))}
    check(int(float(table2["ALL_ITEMS (majority)"]["n"])) == 109, f"n with empty-excludes={table2['ALL_ITEMS (majority)']['n']}")

    print("OK label filter (disagreement excluded, O-none kept, empty warned)")


def test_frozen_prompt_and_schema() -> None:
    """The prompt is the one of the reported run, and its schema is ExpertIntent's."""
    prompt = load_system_prompt()
    check(prompt_sha256(prompt) == FROZEN_PROMPT_SHA256, "data/system_prompt.txt hash differs from the reported run")
    archived = ARCHIVE_DIR / "raw" / "prompt_used.txt"
    if archived.is_file():
        check(load_system_prompt(archived) == prompt, "data/system_prompt.txt differs from results/raw/prompt_used.txt")
    meta_path = ARCHIVE_DIR / "raw" / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        check(meta.get("prompt_sha256") == FROZEN_PROMPT_SHA256, "run_meta.json prompt hash differs")
    # The schema block that LangChain's PydanticOutputParser put into the prompt
    # is ExpertIntent.model_json_schema() without the top-level title/type.
    start = prompt.index("Here is the output schema:\n```\n") + len("Here is the output schema:\n```\n")
    end = prompt.index("\n```", start)
    embedded = json.loads(prompt[start:end])
    schema = ExpertIntent.model_json_schema()
    schema.pop("title", None)
    schema.pop("type", None)
    check(embedded == schema, "ExpertIntent schema differs from the one in the frozen prompt")
    print("OK frozen prompt (SHA-256 matches the run) and ExpertIntent schema")


def test_reparse_archived_outputs() -> None:
    """Re-parsing the 333 saved raw replies gives the archived codes and error types."""
    path = ARCHIVE_DIR / "raw" / "intent_results.csv"
    if not path.is_file():
        print("SKIP re-parse (no archived results)")
        return
    bundle = build_prompt_bundle(None, allow_modified_prompt=False)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    check(len(rows) == 333, f"archived run should have 333 rows, got {len(rows)}")
    bad = []
    for row in rows:
        codes, err_type, _fields, _merr = parse_one(bundle, row["raw_output"])
        got_codes = "" if codes == "parse_error" else codes
        if err_type == "parse_error":
            ok = row["error_type"] == "parse_error" and row["parsed_codes"] == ""
        else:
            ok = got_codes == row["parsed_codes"]
        if not ok:
            bad.append((row["id"], row["repeat"], row["parsed_codes"], got_codes, err_type))
    check(not bad, f"re-parse mismatch on {len(bad)} rows, e.g. {bad[:3]}")
    print(f"OK re-parse of {len(rows)} archived replies matches parsed_codes")


def test_refuse_archive() -> None:
    bad = subprocess.run(
        [PY, "run_intent_test.py", "--mock", "gold", "--out-dir", str(ARCHIVE_DIR / "raw")],
        cwd=str(HERE), text=True, capture_output=True,
    )
    check(bad.returncode != 0 and "Refusing" in (bad.stdout + bad.stderr), "mock into results/ should be refused")
    bad = subprocess.run(
        [PY, "analyze_intent.py", "--out-dir", str(ARCHIVE_DIR / "analysis")],
        cwd=str(HERE), text=True, capture_output=True,
    )
    check(bad.returncode != 0 and "Refusing" in (bad.stdout + bad.stderr), "analysis into results/ should be refused")
    print("OK archived results/ folder is write-protected")


def main() -> int:
    test_frozen_prompt_and_schema()
    test_reparse_archived_outputs()
    test_refuse_archive()
    test_mapping_roundtrip()
    test_wilson()
    test_mcnemar()
    test_error_types()
    test_label_helpers()
    test_mock_e2e()
    test_label_filter()
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
