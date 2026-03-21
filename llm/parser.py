"""Parse and validate LLM JSON responses for the diagnostic replay task."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedResponse:
    """Structured result from parsing an LLM response."""

    assessment: str | None = None
    differential: list[dict] | None = None  # [{"diagnosis": str, "confidence": float}]
    key_findings: list[int] | None = None
    recommended_actions: list[dict] | None = None  # [{"action": str, "detail": str}]
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
    """Parse LLM response text and validate required fields.

    Never raises — sets parse_error on failure.
    """
    raw = parse_llm_json(text)
    if raw is None:
        return ParsedResponse(parse_error="Could not extract JSON from response")

    if not isinstance(raw, dict):
        return ParsedResponse(parse_error=f"Expected JSON object, got {type(raw).__name__}")

    errors = []

    # Extract and validate assessment
    assessment = raw.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        errors.append("missing or empty 'assessment'")
        assessment = None

    # Extract and validate differential
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

    # Extract key_findings
    key_findings = raw.get("key_findings")
    if isinstance(key_findings, list):
        key_findings = [x for x in key_findings if isinstance(x, int)]
        if not key_findings:
            key_findings = None
    else:
        key_findings = None

    # Extract recommended_actions
    recommended_actions = raw.get("recommended_actions")
    if isinstance(recommended_actions, list):
        valid_actions = []
        for entry in recommended_actions:
            if isinstance(entry, dict) and "action" in entry:
                valid_actions.append(entry)
        recommended_actions = valid_actions if valid_actions else None
    else:
        recommended_actions = None

    return ParsedResponse(
        assessment=assessment,
        differential=differential,
        key_findings=key_findings,
        recommended_actions=recommended_actions,
        raw_json=raw,
        parse_error="; ".join(errors) if errors else None,
    )
