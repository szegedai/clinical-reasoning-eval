"""Parse and validate LLM JSON responses for the diagnostic replay task."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedResponse:
    """Structured result from parsing an LLM response."""

    assessment: str | None = None
    delta: str | None = None
    differential: list[dict] | None = None  # [{"diagnosis": str, "confidence": float}]
    key_findings: list[int] | None = None
    actions: list[dict] | None = None  # [{"action": str, "detail": str}]
    confident_in_diagnosis: bool | None = None
    raw_json: dict | None = None
    parse_error: str | None = None


def parse_llm_json(text: str) -> dict | None:
    """Extract JSON from LLM text with 3 fallbacks.

    1. Direct JSON parse
    2. Code fence extraction (```json ... ```)
    3. First '{' to last '}'
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Code fence extraction
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_and_validate(text: str) -> ParsedResponse:
    """Parse LLM response text and validate required fields."""
    raw = parse_llm_json(text)
    if raw is None:
        return ParsedResponse(parse_error="Could not extract JSON from response")

    if not isinstance(raw, dict):
        return ParsedResponse(parse_error=f"Expected JSON object, got {type(raw).__name__}")

    errors = []

    assessment = raw.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        errors.append("missing or empty 'assessment'")
        assessment = None

    differential = raw.get("differential")
    if isinstance(differential, list):
        valid_diffs = []
        for i, entry in enumerate(differential):
            if isinstance(entry, dict) and "diagnosis" in entry and "confidence" in entry:
                try:
                    entry["confidence"] = float(entry["confidence"])
                except (TypeError, ValueError):
                    errors.append(f"differential[{i}].confidence not numeric")
                    continue
                valid_diffs.append(entry)
            else:
                errors.append(f"differential[{i}] missing diagnosis/confidence")
        differential = valid_diffs if valid_diffs else None
        if differential is None:
            errors.append("no valid differential entries")
    else:
        errors.append("missing or invalid 'differential'")
        differential = None

    key_findings = raw.get("key_findings")
    if isinstance(key_findings, list):
        key_findings = [x for x in key_findings if isinstance(x, int)]
    else:
        key_findings = None

    delta = raw.get("delta")
    if not isinstance(delta, str) or not delta.strip():
        errors.append(f"missing or empty 'delta' (got {delta!r})")
        delta = None

    actions = raw.get("actions")
    if isinstance(actions, list):
        valid_actions = []
        for entry in actions:
            if isinstance(entry, dict) and "action" in entry:
                valid_actions.append(entry)
        actions = valid_actions if valid_actions else None
    else:
        actions = None

    confident = raw.get("confident_in_diagnosis")
    if not isinstance(confident, bool):
        errors.append(f"invalid or missing 'confident_in_diagnosis' (got {confident!r})")
        confident = None

    return ParsedResponse(
        assessment=assessment,
        delta=delta,
        differential=differential,
        key_findings=key_findings,
        actions=actions,
        confident_in_diagnosis=confident,
        raw_json=raw,
        parse_error="; ".join(errors) if errors else None,
    )
