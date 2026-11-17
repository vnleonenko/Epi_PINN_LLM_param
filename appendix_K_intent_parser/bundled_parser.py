"""ExpertIntent schema, frozen system prompt and reply parser of the Intent Parser.

This module makes the Appendix K folder independent of the repository's
agents/ package. The repository's agents/IntentParserAgent.py is the earlier
peak-only parser (no growth or mortality fields); the test in Appendix K used
the full parser of the article, reproduced here:

- ExpertIntent: the article's schema, with the growth and mortality axes.
  Its JSON schema equals the one embedded in the frozen prompt (checked by
  selftest.py).
- data/system_prompt.txt: the rendered system prompt of the reported run,
  byte for byte (SHA-256 2e89915c...f94bfb997e).
- parse_intent: LangChain's PydanticOutputParser when langchain-core is
  installed (the parser the agent uses); if it is missing or fails, a JSON
  extractor followed by the same Pydantic validation. Re-parsing the 333
  archived replies this way gives the archived codes, with or without
  langchain-core (selftest.py).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schema: the article's ExpertIntent (growth + mortality axes included)
# ---------------------------------------------------------------------------

class ExpertIntent(BaseModel):
    cares_about_position: bool = Field(description="Whether expert cares about peak position")
    position_direction: str = Field(description="Expected direction: 'later', 'earlier', or 'any'")
    cares_about_height: bool = Field(description="Whether expert cares about peak height")
    height_direction: str = Field(description="Expected direction: 'higher', 'lower', or 'any'")
    cares_about_growth: bool = Field(
        default=False,
        description="Whether expert cares about the initial growth rate of the epidemic",
    )
    growth_direction: str = Field(
        default="any",
        description="Expected direction: 'faster', 'slower', or 'any'",
    )
    cares_about_mortality: bool = Field(
        default=False,
        description="Whether expert cares about mortality (share of resolved cases that die)",
    )
    mortality_direction: str = Field(
        default="any",
        description="Expected direction: 'higher', 'lower', or 'any'",
    )
    primary_metric: Literal["position", "height", "both", "growth", "mortality"] = "both"
    reasoning: str = Field(description="Brief parsing reasoning")


# Human message template used by IntentParserAgent._create_prompt().
HUMAN_TEMPLATE = "Expert comment: {expert_comment}"

# The exact system prompt sent to the model in the reported run. It is the
# rendered prompt of the article's IntentParserAgent (prompt body plus
# LangChain's PydanticOutputParser format instructions for ExpertIntent),
# saved byte for byte; results/raw/prompt_used.txt is the copy written
# during the run.
FROZEN_PROMPT_PATH = Path(__file__).resolve().parent / "data" / "system_prompt.txt"
FROZEN_PROMPT_SHA256 = "2e89915c6d13b514aacc0eeb0a912372f7d8d7c1a1b4e044c52620f94bfb997e"


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Read the system prompt exactly as stored (no newline translation)."""
    path = Path(path) if path is not None else FROZEN_PROMPT_PATH
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_messages(expert_comment: str, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
    sp = system_prompt if system_prompt is not None else load_system_prompt()
    return [
        {"role": "system", "content": sp},
        {"role": "user", "content": HUMAN_TEMPLATE.format(expert_comment=expert_comment)},
    ]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> str:
    """Pull a JSON object out of a raw model reply (fences / leading prose)."""
    if text is None:
        raise ValueError("empty model output")
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model output")
    m = _JSON_FENCE.search(stripped)
    if m:
        stripped = m.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return stripped[start : end + 1]


def parse_intent(raw_output: str, model: type[BaseModel] = ExpertIntent) -> Any:
    """Parse raw LLM text into ExpertIntent. Raises ValueError on failure."""
    # Prefer LangChain's PydanticOutputParser when present, as IntentParserAgent does.
    try:
        from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

        parser = PydanticOutputParser(pydantic_object=model)
        try:
            return parser.parse(raw_output)
        except Exception:
            # Fall back to the JSON extractor below, then validate again.
            pass
    except ImportError:
        parser = None

    blob = extract_json_object(raw_output)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    return model.model_validate(data)
