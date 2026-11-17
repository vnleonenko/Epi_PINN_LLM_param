#!/usr/bin/env python3
"""Run the Intent Parser test of Appendix K.

Calls the Intent Parser 3 times on each of the 111 corpus items and
writes intent_results.csv. Talks to LM Studio's OpenAI-compatible API, or
to a --mock LLM for a dry run that does not need the model.

The system prompt is data/system_prompt.txt, byte-identical to the prompt of
the reported run (its SHA-256 is checked), and replies are parsed with the
ExpertIntent schema in bundled_parser.py. The repository's agents/ package is
not imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# MAPPING comes from intent_mapping.py, which was frozen before the run
# (documented in mapping.md).
from intent_mapping import (  # noqa: E402
    MAPPING,
    PARSE_ERROR,
    codes_to_intent_dict,
    format_codes,
    is_exact_match,
    map_intent_to_codes,
    parse_codes,
    classify_error,
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"
RESULT_FIELDS = [
    "id",
    "repeat",
    "raw_output",
    "parsed_codes",
    "correct",
    "error_type",
    "latency_s",
    "level",
    "gold",
    "group",
    "parsed_fields",
    "mapping_error",
]


# ---------------------------------------------------------------------------
# Prompt loading (frozen prompt + bundled schema; no repository import)
# ---------------------------------------------------------------------------

REQUIRED_INTENT_FIELDS = [
    "cares_about_growth",
    "growth_direction",
    "cares_about_mortality",
    "mortality_direction",
]


class PromptBundle:
    """Unified access to the system prompt, message rendering and parser."""

    def __init__(
        self,
        *,
        expert_intent_cls: type,
        system_prompt: str,
        render_messages,
        parse_raw,
        source: str,
        source_extra: Optional[dict] = None,
    ):
        self.expert_intent_cls = expert_intent_cls
        self.system_prompt = system_prompt
        self.render_messages = render_messages
        self.parse_raw = parse_raw
        self.source = source
        self.source_extra = source_extra or {}

    def sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()


def build_prompt_bundle(prompt_file: Optional[Path], *, allow_modified_prompt: bool) -> PromptBundle:
    """Frozen system prompt from data/system_prompt.txt and the bundled parser.

    The prompt is sent byte for byte. A prompt whose SHA-256 differs from the
    reported run is refused unless --allow-modified-prompt is given.
    """
    from bundled_parser import (
        FROZEN_PROMPT_PATH,
        FROZEN_PROMPT_SHA256,
        HUMAN_TEMPLATE,
        ExpertIntent,
        load_system_prompt,
        parse_intent,
        prompt_sha256,
    )

    path = Path(prompt_file) if prompt_file is not None else FROZEN_PROMPT_PATH
    if not path.is_file():
        raise SystemExit(f"system prompt not found: {path}")
    system_prompt = load_system_prompt(path)
    digest = prompt_sha256(system_prompt)
    if digest != FROZEN_PROMPT_SHA256:
        msg = (
            f"system prompt {path} has SHA-256 {digest}, the reported run used "
            f"{FROZEN_PROMPT_SHA256}."
        )
        if not allow_modified_prompt:
            raise SystemExit(msg + " Pass --allow-modified-prompt to run it anyway.")
        print("WARNING: " + msg, file=sys.stderr)

    def render_messages(comment: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": HUMAN_TEMPLATE.format(expert_comment=comment)},
        ]

    try:
        rel = path.resolve().relative_to(HERE)
        shown = str(rel)
    except ValueError:
        shown = path.name
    return PromptBundle(
        expert_intent_cls=ExpertIntent,
        system_prompt=system_prompt,
        render_messages=render_messages,
        parse_raw=lambda text: parse_intent(text, model=ExpertIntent),
        source=f"frozen:{shown}",
        source_extra={"prompt_matches_reported_run": digest == FROZEN_PROMPT_SHA256},
    )


# ---------------------------------------------------------------------------
# Corpus / keyword baseline
# ---------------------------------------------------------------------------

DATA_DIR = HERE / "data"
ARCHIVE_DIR = HERE / "results"  # archived run, never written to


def rerun_root() -> Path:
    """Output root for reruns: $INTENT_OUT, else <folder>/results_rerun."""
    env = os.environ.get("INTENT_OUT")
    return Path(env).expanduser().resolve() if env else HERE / "results_rerun"


def default_corpus_path() -> Path:
    return DATA_DIR / "intent_corpus.csv"


def default_baseline_path() -> Path:
    return DATA_DIR / "baseline_results.csv"


def default_keyword_py() -> Path:
    return HERE / "keyword_baseline.py"


def load_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty corpus: {path}")
    for required in ("id", "text", "gold"):
        if required not in rows[0]:
            raise SystemExit(f"corpus {path} is missing column {required}")
    if len(rows) != 111:
        print(f"WARNING: corpus has {len(rows)} rows, protocol expects 111", file=sys.stderr)
    return rows


def load_keyword_module(path: Path):
    spec = importlib.util.spec_from_file_location("keyword_baseline", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import keyword baseline from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_keyword_codes(baseline_csv: Path, keyword_py: Path, corpus: list[dict]) -> dict[str, frozenset]:
    """Map item id to keyword-rule codes: from the frozen CSV if present, else by running the rule."""
    out: dict[str, frozenset] = {}
    if baseline_csv.is_file():
        with baseline_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                col = row.get("keyword_rule") or row.get("keyword") or ""
                out[row["id"]] = parse_codes(col)
        return out
    mod = load_keyword_module(keyword_py)
    for row in corpus:
        out[row["id"]] = frozenset(mod.parse(row["text"]))
    return out


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def intent_to_json_text(intent_dict: dict) -> str:
    return json.dumps(intent_dict, ensure_ascii=False, indent=2)


def mixed_codes_for(item: dict, repeat: int, gold: frozenset) -> tuple[Optional[frozenset], bool]:
    """Deterministic mixed mock: (codes or None, parse_failed)."""
    item_id = item["id"]
    # All three repeats parse-fail.
    if item_id == "Q002":
        return None, True
    # Opposite direction, unanimous.
    if item_id == "Q013":
        return frozenset({"peak_lower"}), False  # gold peak_higher
    # Wrong axis, unanimous.
    if item_id == "Q027":
        return frozenset({"peak_higher"}), False  # gold peak_earlier
    # Extra target on a held axis.
    if item_id == "Q098":
        return frozenset({"peak_lower", "peak_later"}), False  # gold peak_lower
    # Missing one of two compound targets.
    if item_id == "Q091":
        return frozenset({"peak_higher"}), False  # gold peak_higher;peak_earlier
    # False positive on N.
    if item_id == "Q100":
        return frozenset({"peak_higher"}), False
    # One inconsistent repeat (majority still gold).
    if item_id == "Q005" and repeat == 3:
        return frozenset({"growth_slower"}), False
    # Group O: opposite (critical).
    if item_id == "Q108":
        return frozenset({"growth_faster"}), False
    # Group O: "not supported" as none (correct under the group-O rule, not a strict match).
    if item_id == "Q109":
        return frozenset({"none"}), False
    return gold, False


class MockLLM:
    def __init__(self, mode: str, keyword_codes: Optional[dict[str, frozenset]] = None):
        if mode not in {"gold", "keyword", "mixed"}:
            raise ValueError(f"unknown mock mode: {mode}")
        self.mode = mode
        self.keyword_codes = keyword_codes or {}

    def complete(self, item: dict, repeat: int, gold: frozenset) -> tuple[str, float]:
        t0 = time.perf_counter()
        parse_failed = False
        if self.mode == "gold":
            codes = gold
        elif self.mode == "keyword":
            codes = self.keyword_codes.get(item["id"], frozenset({"none"}))
        else:
            codes, parse_failed = mixed_codes_for(item, repeat, gold)
        if parse_failed:
            raw = "NOT JSON: the mock deliberately returns a parse failure."
        else:
            intent = codes_to_intent_dict(codes, reasoning=f"mock:{self.mode}:{item['id']}:r{repeat}")
            raw = intent_to_json_text(intent)
        # Tiny synthetic latency so the column is populated and resume tests work.
        latency = time.perf_counter() - t0 + 0.0001
        return raw, latency


def lmstudio_models(base_url: str, timeout: float) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer lm-studio"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id") or m.get("name") for m in data.get("data", []) if m]
    except Exception as exc:
        print(f"WARNING: could not list models at {url}: {exc}", file=sys.stderr)
        return []


def call_lmstudio(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
    timeout: float,
) -> tuple[str, float, dict]:
    """POST /chat/completions. Prefer the openai package, else stdlib urllib."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    t0 = time.perf_counter()
    meta = {"client": "urllib", "seed_sent": seed is not None}

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)
        kwargs = dict(payload)
        # Some servers reject unknown fields; seed is optional.
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            if "seed" in kwargs:
                kwargs.pop("seed")
                meta["seed_dropped"] = True
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        latency = time.perf_counter() - t0
        content = resp.choices[0].message.content or ""
        meta["client"] = "openai"
        meta["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        return content, latency, meta
    except ImportError:
        pass

    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        # Retry without seed if the server rejects it.
        if seed is not None and "seed" in payload:
            payload.pop("seed")
            meta["seed_dropped"] = True
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer lm-studio",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        else:
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
    latency = time.perf_counter() - t0
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected LM Studio response: {body!r}") from exc
    return content, latency, meta


# ---------------------------------------------------------------------------
# Results I/O (resume-safe)
# ---------------------------------------------------------------------------

def read_done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["id"], int(row["repeat"])))
            except (KeyError, ValueError):
                continue
    return done


def open_results_writer(path: Path, resume: bool):
    exists = path.is_file() and path.stat().st_size > 0
    if exists and resume:
        f = path.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        return f, writer
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    return f, writer


def parse_one(bundle: PromptBundle, raw: str) -> tuple[str, str, str, str]:
    """Returns (parsed_codes, error_type_if_fail, parsed_fields_json, mapping_error)."""
    try:
        intent = bundle.parse_raw(raw)
    except Exception as exc:
        return PARSE_ERROR, "parse_error", "", f"parse: {exc}"
    if hasattr(intent, "model_dump"):
        fields = intent.model_dump()
    elif hasattr(intent, "dict"):
        fields = intent.dict()
    elif isinstance(intent, dict):
        fields = intent
    else:
        fields = {
            k: getattr(intent, k)
            for k in (
                "cares_about_position",
                "position_direction",
                "cares_about_height",
                "height_direction",
                "cares_about_growth",
                "growth_direction",
                "cares_about_mortality",
                "mortality_direction",
                "primary_metric",
                "reasoning",
            )
            if hasattr(intent, k)
        }
    codes, mapping_error = map_intent_to_codes(intent)
    if mapping_error:
        return PARSE_ERROR, "parse_error", json.dumps(fields, ensure_ascii=False), mapping_error
    return format_codes(codes), "", json.dumps(fields, ensure_ascii=False), ""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def pkg_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "present")
    except Exception:
        return None


def write_prompt_and_meta(
    out_dir: Path,
    bundle: PromptBundle,
    args: argparse.Namespace,
    extra: dict,
):
    prompt_path = out_dir / "prompt_used.txt"
    prompt_path.write_text(bundle.system_prompt, encoding="utf-8")
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "base_url": None if args.mock else args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "seed": args.seed,
        "lmstudio_version": args.lmstudio_version,
        "model_format": args.model_format,
        "mock": args.mock,
        "prompt_source": bundle.source,
        "prompt_sha256": bundle.sha256(),
        "prompt_path": str(prompt_path),
        "mapping": {f"{a}/{d}": c for (a, d), c in MAPPING.items()},
        "python": sys.version,
        "versions": {
            "pydantic": pkg_version("pydantic"),
            "langchain_core": pkg_version("langchain_core"),
            "langchain_openai": pkg_version("langchain_openai"),
            "openai": pkg_version("openai"),
        },
        **bundle.source_extra,
        **extra,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def load_existing_meta(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        out = args.out_dir.expanduser().resolve()
    elif args.mock:
        out = (rerun_root() / "mock").resolve()
    else:
        out = (rerun_root() / "raw").resolve()

    arch = ARCHIVE_DIR.resolve()
    if out == arch or arch in out.parents:
        raise SystemExit(
            f"Refusing to write into the archived results folder {arch}. "
            "Pass --out-dir or set INTENT_OUT."
        )
    if args.mock and out == HERE.resolve():
        raise SystemExit(
            "Refusing to write mock results into the experiment folder. "
            "Pass --out-dir pointing at a dedicated directory "
            "(default for --mock is results_rerun/mock/)."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(args: argparse.Namespace) -> int:
    if args.print_prompt:
        bundle = build_prompt_bundle(args.prompt_file, allow_modified_prompt=args.allow_modified_prompt)
        sys.stdout.write(bundle.system_prompt)
        if not bundle.system_prompt.endswith("\n"):
            sys.stdout.write("\n")
        print(f"\n# source: {bundle.source}", file=sys.stderr)
        print(f"# sha256: {bundle.sha256()}", file=sys.stderr)
        return 0
    out_dir = resolve_out_dir(args)
    corpus = load_corpus(args.corpus)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        corpus = [r for r in corpus if r["id"] in wanted]
        if not corpus:
            raise SystemExit(f"no corpus rows match --ids {wanted}")

    bundle = build_prompt_bundle(args.prompt_file, allow_modified_prompt=args.allow_modified_prompt)

    results_path = out_dir / "intent_results.csv"
    existing_meta = load_existing_meta(out_dir / "run_meta.json")
    done = read_done_keys(results_path) if args.resume else set()
    if done and existing_meta:
        prev_hash = existing_meta.get("prompt_sha256")
        if prev_hash and prev_hash != bundle.sha256():
            raise SystemExit(
                "Refusing to resume: prompt_used.txt / run_meta.json sha256 "
                f"{prev_hash} does not match the prompt loaded now "
                f"({bundle.sha256()}). Start a new --out-dir or restore the "
                "original prompt."
            )
        prev_model = existing_meta.get("model")
        if prev_model and prev_model != args.model and not args.mock:
            raise SystemExit(
                f"Refusing to resume: previous model was {prev_model}, "
                f"this run is {args.model}."
            )

    keyword_codes = None
    if args.mock == "keyword":
        keyword_codes = load_keyword_codes(args.baseline, args.keyword_baseline, corpus)

    mock = MockLLM(args.mock, keyword_codes) if args.mock else None
    server_models: list[str] = []
    if not mock:
        server_models = lmstudio_models(args.base_url, args.timeout)
        if not server_models:
            raise SystemExit(
                f"No model server answered at {args.base_url}. Start the LM Studio "
                "local server (or use --mock for a dry run)."
            )
        if args.model not in server_models:
            print(
                f"WARNING: model {args.model!r} is not in the LM Studio list "
                f"{server_models}. The server may still alias it; continuing.",
                file=sys.stderr,
            )

    meta = write_prompt_and_meta(
        out_dir,
        bundle,
        args,
        extra={
            "corpus": str(args.corpus.resolve()),
            "out_dir": str(out_dir),
            "results_path": str(results_path),
            "resume": bool(done),
            "n_already_done": len(done),
            "lmstudio_models": server_models,
            "status": "running",
        },
    )
    print(f"prompt source: {bundle.source}")
    print(f"prompt sha256: {bundle.sha256()}")
    print(f"writing: {results_path}")

    work = []
    for row in corpus:
        for rpt in range(1, args.repeats + 1):
            work.append((row, rpt))
    if args.limit is not None:
        # Limit applies to remaining work, but keep already-done rows skipped.
        remaining = [(r, i) for r, i in work if (r["id"], i) not in done]
        keep = set((r["id"], i) for r, i in remaining[: args.limit])
        work = [(r, i) for r, i in work if (r["id"], i) in done or (r["id"], i) in keep]

    n_total = len(work)
    n_todo = sum(1 for r, i in work if (r["id"], i) not in done)
    print(f"planned calls: {n_total} ({n_todo} remaining, {len(done)} already done)")

    f, writer = open_results_writer(results_path, resume=bool(args.resume and results_path.is_file()))
    n_written = 0
    n_failed = 0
    seed_dropped_any = False
    t_run = time.perf_counter()
    try:
        for idx, (row, rpt) in enumerate(work, start=1):
            key = (row["id"], rpt)
            if key in done:
                continue
            gold = parse_codes(row["gold"])
            gold_s = format_codes(gold)
            group = row.get("group", "")
            level = row.get("level", "")
            raw = ""
            latency = float("nan")
            call_error = ""
            try:
                if mock is not None:
                    raw, latency = mock.complete(row, rpt, gold)
                else:
                    messages = bundle.render_messages(row["text"])
                    raw, latency, _call_meta = call_lmstudio(
                        base_url=args.base_url,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                        timeout=args.timeout,
                    )
                    if isinstance(_call_meta, dict) and _call_meta.get("seed_dropped"):
                        seed_dropped_any = True
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                call_error = f"call: {exc}"
                raw = f"[CALL_ERROR] {exc}\n{traceback.format_exc()}"
                latency = latency if latency == latency else float("nan")
                n_failed += 1

            if call_error:
                parsed_codes, err_type, fields_json, mapping_error = (
                    PARSE_ERROR,
                    "parse_error",
                    "",
                    call_error,
                )
            else:
                parsed_codes, err_type, fields_json, mapping_error = parse_one(bundle, raw)
            pred = parse_codes(parsed_codes) if parsed_codes != PARSE_ERROR else frozenset({PARSE_ERROR})
            if not err_type:
                err_type, _flags = classify_error(
                    pred, gold, group, level, parse_failed=False, out_of_set_none_ok=False
                )
            correct = int(
                parsed_codes != PARSE_ERROR and is_exact_match(pred, gold)
            )
            writer.writerow(
                {
                    "id": row["id"],
                    "repeat": rpt,
                    "raw_output": raw,
                    "parsed_codes": parsed_codes if parsed_codes != PARSE_ERROR else "",
                    "correct": correct,
                    "error_type": err_type,
                    "latency_s": f"{latency:.4f}" if latency == latency else "",
                    "level": level,
                    "gold": gold_s,
                    "group": group,
                    "parsed_fields": fields_json,
                    "mapping_error": mapping_error,
                }
            )
            f.flush()
            n_written += 1
            if n_written % 10 == 0 or n_written == n_todo:
                elapsed = time.perf_counter() - t_run
                rate = n_written / elapsed if elapsed else 0
                left = n_todo - n_written
                eta = left / rate if rate else 0
                print(
                    f"  {n_written}/{n_todo} written  "
                    f"last={row['id']} r{rpt} codes={parsed_codes or PARSE_ERROR}  "
                    f"elapsed={elapsed:.1f}s eta={eta:.0f}s"
                )
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results were flushed; rerun the same command to resume.")
        meta_now = load_existing_meta(out_dir / "run_meta.json") or dict(meta)
        meta_now.update(
            {
                "status": "interrupted",
                "n_written_this_session": n_written,
                "ended": datetime.now().astimezone().isoformat(),
            }
        )
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta_now, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 130
    finally:
        f.close()

    elapsed = time.perf_counter() - t_run
    final_done = read_done_keys(results_path)
    expected_keys = {(r["id"], i) for r, i in work}
    complete = expected_keys <= final_done and args.limit is None and not args.ids
    meta_now = load_existing_meta(out_dir / "run_meta.json") or meta
    meta_now.update(
        {
            "status": "complete" if expected_keys <= final_done else "partial",
            "n_written_this_session": n_written,
            "n_call_failures_this_session": n_failed,
            "n_rows_in_csv": len(final_done),
            "elapsed_s": round(elapsed, 3),
            "ended": datetime.now().astimezone().isoformat(),
            "complete_for_this_plan": bool(expected_keys <= final_done),
            "seed_dropped": seed_dropped_any,
            "fully_complete_333": complete and args.repeats == 3 and len(final_done) == 333,
        }
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta_now, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"done: wrote {n_written} new rows in {elapsed:.1f}s; "
        f"csv now has {len(final_done)} rows; status={meta_now['status']}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="System prompt to send. Default: data/system_prompt.txt (the prompt of the reported run).",
    )
    p.add_argument("--corpus", type=Path, default=default_corpus_path(), help="intent_corpus.csv")
    p.add_argument("--baseline", type=Path, default=default_baseline_path(), help="baseline_results.csv (for --mock keyword)")
    p.add_argument("--keyword-baseline", type=Path, default=default_keyword_py(), help="keyword_baseline.py")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: results_rerun/raw/ (real run) or results_rerun/mock/ (--mock); with INTENT_OUT set, $INTENT_OUT/raw or $INTENT_OUT/mock. The archived results/ folder is never written to.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LM Studio OpenAI-compatible base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name as served by LM Studio")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048, help="2048 as in Table 2 runs")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Sent as the OpenAI 'seed' field and logged; if the server rejects it, it is dropped and seed_dropped=true is logged. Use -1 to not send a seed.")
    p.add_argument("--lmstudio-version", default=None, help="LM Studio version as shown in the app, e.g. 0.4.26+4 (logged)")
    p.add_argument("--model-format", default=None, help="Model build as shown in LM Studio, e.g. 'MLX 4-bit' (logged)")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--mock",
        nargs="?",
        const="gold",
        choices=["gold", "keyword", "mixed"],
        default=None,
        help="Do not call LM Studio. gold=perfect labels; keyword=frozen keyword rule; mixed=injected error types. Writes to results_rerun/mock/ unless --out-dir is set.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max new (id, repeat) pairs to run (for smoke tests / resume checks)")
    p.add_argument("--ids", default=None, help="Comma-separated item ids to run (e.g. Q001,Q002)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip (id, repeat) already in intent_results.csv (default)")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Overwrite intent_results.csv and start from scratch")
    p.add_argument(
        "--allow-modified-prompt",
        action="store_true",
        help="Run even if the prompt's SHA-256 differs from the reported run (not the reported setting).",
    )
    p.add_argument("--print-prompt", action="store_true", help="Print the rendered system prompt and exit")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.seed is not None and args.seed < 0:
        args.seed = None
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
