"""Pre-registered Intent Parser output → target-code mapping.

Frozen before the run. Do not change after seeing model outputs.
primary_metric and reasoning are ignored.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Mapping table (also documented in mapping.md)
# ---------------------------------------------------------------------------
# A code is emitted only when cares_X is true AND the direction is a concrete
# value listed below. Direction "any" (or cares_X false) contributes no code.
# If the resulting set is empty, it is recorded as {"none"}.

MAPPING = {
    ("position", "later"): "peak_later",
    ("position", "earlier"): "peak_earlier",
    ("height", "higher"): "peak_higher",
    ("height", "lower"): "peak_lower",
    ("growth", "faster"): "growth_faster",
    ("growth", "slower"): "growth_slower",
    ("mortality", "higher"): "mortality_higher",
    ("mortality", "lower"): "mortality_lower",
}

# Axis → (cares_field, direction_field) on ExpertIntent
AXIS_FIELDS = {
    "position": ("cares_about_position", "position_direction"),
    "height": ("cares_about_height", "height_direction"),
    "growth": ("cares_about_growth", "growth_direction"),
    "mortality": ("cares_about_mortality", "mortality_direction"),
}

VALID_DIRECTIONS = {
    "position": {"later", "earlier", "any"},
    "height": {"higher", "lower", "any"},
    "growth": {"faster", "slower", "any"},
    "mortality": {"higher", "lower", "any"},
}

OPPOSITE = {
    "peak_higher": "peak_lower",
    "peak_lower": "peak_higher",
    "peak_earlier": "peak_later",
    "peak_later": "peak_earlier",
    "growth_faster": "growth_slower",
    "growth_slower": "growth_faster",
    "mortality_higher": "mortality_lower",
    "mortality_lower": "mortality_higher",
}

AXIS_OF = {
    "peak_higher": "height",
    "peak_lower": "height",
    "peak_earlier": "position",
    "peak_later": "position",
    "growth_faster": "growth",
    "growth_slower": "growth",
    "mortality_higher": "mortality",
    "mortality_lower": "mortality",
}

# Six targets tested in the paper, plus none. Used for the 7×7 matrix.
MATRIX_LABELS = [
    "peak_higher",
    "peak_lower",
    "peak_earlier",
    "peak_later",
    "growth_faster",
    "mortality_higher",
    "none",
]

# Level order as in the TASK document.
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "RU", "C", "H", "N", "O"]

# Sentinel used when a repeat cannot be parsed.
PARSE_ERROR = "__parse_error__"

# Gold / parsed-code string separator used in the corpus and in outputs.
# The TASK typography looks like ";;" in one place; the CSV and the frozen
# keyword baseline both use a single ';'.
CODE_SEP = ";"


def parse_codes(value: Any) -> frozenset[str]:
    """Parse a gold or predicted code field into a set.

    Separator is ';'. Empty / missing becomes {'none'}.
    """
    if value is None:
        return frozenset({"none"})
    text = str(value).strip()
    if not text:
        return frozenset({"none"})
    if text in {PARSE_ERROR, "parse_error"}:
        return frozenset({PARSE_ERROR})
    parts = [p.strip() for p in text.split(CODE_SEP) if p.strip()]
    return frozenset(parts) if parts else frozenset({"none"})


def format_codes(codes: Iterable[str]) -> str:
    codes_set = frozenset(codes)
    if not codes_set or codes_set == frozenset({"none"}):
        return "none"
    if PARSE_ERROR in codes_set and codes_set <= {PARSE_ERROR, "none"}:
        return "parse_error"
    meaningful = {c for c in codes_set if c != "none" and c != PARSE_ERROR}
    if not meaningful:
        return "none"
    return CODE_SEP.join(sorted(meaningful))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0", "", "none", "null"}:
        return False
    raise ValueError(f"cannot coerce {value!r} to bool")


def _get_field(intent: Any, name: str, default: Any = None) -> Any:
    if intent is None:
        return default
    if isinstance(intent, Mapping):
        return intent.get(name, default)
    return getattr(intent, name, default)


def map_intent_to_codes(intent: Any) -> tuple[frozenset[str], Optional[str]]:
    """Map a parsed ExpertIntent (object or dict) to a code set.

    Returns (codes, mapping_error). mapping_error is a short reason if a
    cares_X flag is true and the direction is not in the pre-registered
    table and is not 'any' — the caller should treat that as parse_error.
    """
    codes: set[str] = set()
    for axis, (cares_field, dir_field) in AXIS_FIELDS.items():
        try:
            cares = _as_bool(_get_field(intent, cares_field, False))
        except ValueError as exc:
            return frozenset({PARSE_ERROR}), f"bad bool on {cares_field}: {exc}"
        raw_dir = _get_field(intent, dir_field, "any")
        direction = (str(raw_dir).strip().lower() if raw_dir is not None else "any")
        if not cares:
            continue
        if direction == "any" or direction == "":
            continue
        code = MAPPING.get((axis, direction))
        if code is None:
            allowed = ", ".join(sorted(VALID_DIRECTIONS[axis]))
            return (
                frozenset({PARSE_ERROR}),
                f"unmapped direction {direction!r} on axis {axis} (allowed: {allowed})",
            )
        codes.add(code)
    if not codes:
        return frozenset({"none"}), None
    return frozenset(codes), None


def codes_to_intent_dict(codes: Iterable[str], reasoning: str = "") -> dict:
    """Inverse of map_intent_to_codes, used by the mock LLM.

    primary_metric is a dummy ('both' or the single axis if unique).
    """
    codes_set = {c for c in codes if c and c != "none"}
    out = {
        "cares_about_position": False,
        "position_direction": "any",
        "cares_about_height": False,
        "height_direction": "any",
        "cares_about_growth": False,
        "growth_direction": "any",
        "cares_about_mortality": False,
        "mortality_direction": "any",
        "primary_metric": "both",
        "reasoning": reasoning or "mock",
    }
    axes_used: list[str] = []
    inverse = {code: key for key, code in MAPPING.items()}
    for code in codes_set:
        if code not in inverse:
            continue
        axis, direction = inverse[code]
        cares_field, dir_field = AXIS_FIELDS[axis]
        out[cares_field] = True
        out[dir_field] = direction
        axes_used.append(axis)
    if len(axes_used) == 1:
        out["primary_metric"] = axes_used[0]
    elif len(axes_used) > 1:
        out["primary_metric"] = "both"
    return out


def is_exact_match(pred: Iterable[str], gold: Iterable[str]) -> bool:
    return frozenset(pred) == frozenset(gold)


def is_correct(
    pred: Iterable[str],
    gold: Iterable[str],
    group: str = "",
    level: str = "",
    *,
    out_of_set_none_ok: bool = True,
) -> bool:
    """Exact set match, with the group-O special rule.

    For out_of_set (level O): the gold direction itself OR 'unsupported'
    (mapped as {'none'}) is counted correct. The opposite direction is not.
    """
    pred_set = frozenset(pred)
    gold_set = frozenset(gold)
    if PARSE_ERROR in pred_set:
        return False
    if pred_set == gold_set:
        return True
    if out_of_set_none_ok and _is_out_of_set(group, level) and pred_set == frozenset({"none"}):
        return True
    return False


def is_lenient_match(
    pred: Iterable[str],
    gold: Iterable[str],
    group: str = "",
    level: str = "",
) -> bool:
    """gold ⊆ predicted, treating {'none'} as the empty request.

    For group O, {'none'} (unsupported) is also accepted.
    """
    pred_set = frozenset(pred)
    gold_set = frozenset(gold)
    if PARSE_ERROR in pred_set:
        return False
    if _is_out_of_set(group, level) and pred_set == frozenset({"none"}):
        return True
    pred_t = pred_set - {"none"}
    gold_t = gold_set - {"none"}
    if not gold_t:
        return not pred_t
    return gold_t <= pred_t


def _is_out_of_set(group: str, level: str) -> bool:
    g = (group or "").strip().lower()
    return g in {"out_of_set", "o"} or (level or "").strip().upper() == "O"


def _targets(codes: Iterable[str]) -> frozenset[str]:
    return frozenset(c for c in codes if c not in {"none", PARSE_ERROR})


def classify_error(
    pred: Iterable[str],
    gold: Iterable[str],
    group: str = "",
    level: str = "",
    *,
    parse_failed: bool = False,
    out_of_set_none_ok: bool = True,
) -> tuple[str, list[str]]:
    """Return (primary_error_type, all_flags).

    Priority of the primary type:
      parse_error, opposite_direction, false_positive, wrong_axis,
      extra_target, missing_target, other. Empty string if correct.
    """
    if parse_failed or PARSE_ERROR in set(pred):
        return "parse_error", ["parse_error"]

    pred_set = frozenset(pred)
    gold_set = frozenset(gold)
    if is_correct(pred_set, gold_set, group, level, out_of_set_none_ok=out_of_set_none_ok):
        return "", []

    flags: list[str] = []
    pred_t = _targets(pred_set)
    gold_t = _targets(gold_set)

    if any(OPPOSITE.get(c) in gold_t for c in pred_t):
        flags.append("opposite_direction")

    if not gold_t and pred_t:
        flags.append("false_positive")

    gold_axes = {AXIS_OF[c] for c in gold_t if c in AXIS_OF}
    pred_axes = {AXIS_OF[c] for c in pred_t if c in AXIS_OF}
    extra = pred_t - gold_t
    missing = gold_t - pred_t

    if extra and gold_t and pred_axes and gold_axes:
        # wrong_axis = substitution onto a different axis ("instead of"),
        # not merely an added target on top of a complete gold set.
        extra_off_axis = any(
            AXIS_OF.get(c) not in gold_axes for c in extra if c in AXIS_OF
        )
        if extra_off_axis and (missing or pred_axes.isdisjoint(gold_axes)):
            flags.append("wrong_axis")

    if extra and "false_positive" not in flags:
        # extra on the same axis as gold but not the opposite is unusual;
        # still record it. Opposite extras already have opposite_direction.
        if extra - {OPPOSITE[c] for c in gold_t if c in OPPOSITE}:
            flags.append("extra_target")
        elif "opposite_direction" not in flags:
            flags.append("extra_target")

    if missing:
        flags.append("missing_target")

    # Deduplicate, keep a stable order
    order = [
        "opposite_direction",
        "false_positive",
        "wrong_axis",
        "extra_target",
        "missing_target",
    ]
    seen: list[str] = []
    for name in order:
        if name in flags and name not in seen:
            seen.append(name)
    for name in flags:
        if name not in seen:
            seen.append(name)

    primary = seen[0] if seen else "other"
    return primary, seen


def majority_vote(repeats: Sequence[Iterable[str]]) -> tuple[Optional[frozenset[str]], str]:
    """Majority vote over repeat code sets.

    Returns (codes_or_None, status) where status is
    'majority' (≥2 identical), 'unanimous' (3/3, or all if n<3 and all equal),
    'no_majority' (all differ / no set reaches 2).
    """
    normalized = []
    for item in repeats:
        s = frozenset(item)
        normalized.append(s)
    if not normalized:
        return None, "no_majority"
    counts: dict[frozenset[str], int] = {}
    for s in normalized:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], format_codes(kv[0])))
    n = len(normalized)
    if best[1] >= 2 or (n < 2 and best[1] == n):
        status = "unanimous" if best[1] == n else "majority"
        return best[0], status
    if n == 1:
        return normalized[0], "unanimous"
    return None, "no_majority"


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi


def exact_mcnemar(llm_ok: Sequence[bool], baseline_ok: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test (binomial on discordant pairs).

    b = LLM correct, baseline wrong; c = LLM wrong, baseline correct.
    """
    if len(llm_ok) != len(baseline_ok):
        raise ValueError("paired samples must have the same length")
    b = c = 0
    n_pairs = 0
    for x, y in zip(llm_ok, baseline_ok):
        n_pairs += 1
        if x and not y:
            b += 1
        elif y and not x:
            c += 1
    n = b + c
    if n == 0:
        return {
            "b_llm_only": b,
            "c_baseline_only": c,
            "n_discordant": 0,
            "n_pairs": n_pairs,
            "p_value": 1.0,
        }
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p = 2.0 * tail / (2 ** n)
    if p > 1.0:
        p = 1.0
    return {
        "b_llm_only": b,
        "c_baseline_only": c,
        "n_discordant": n,
        "n_pairs": n_pairs,
        "p_value": p,
    }


def matrix_bucket(codes: Iterable[str]) -> str:
    """Map a code set to a 7×7 (plus 'other') row/column label."""
    s = frozenset(codes)
    if PARSE_ERROR in s:
        return "other"
    if s == frozenset({"none"}) or not _targets(s):
        return "none"
    t = _targets(s)
    if len(t) == 1:
        only = next(iter(t))
        if only in MATRIX_LABELS:
            return only
        return "other"
    return "other"
