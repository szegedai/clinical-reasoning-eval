"""Parse and validate LLM responses for the bias-resilience schema.

{
  "evidence_summary": "<facts only>",
  "working_diagnosis": "<1-2 sentence belief justification>",
  "differential": [{"diagnosis": str, "confidence": float}, ...]  // 5 entries
}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedStep:
    evidence_summary: str | None = None
    working_diagnosis: str | None = None
    differential: list[dict] | None = None  # [{"diagnosis": str, "confidence": float}]
    parse_error: str | None = None
    raw_json: dict | None = None

    def to_dict(self) -> dict:
        return {
            "evidence_summary": self.evidence_summary,
            "working_diagnosis": self.working_diagnosis,
            "differential": self.differential,
            "parse_error": self.parse_error,
        }

    def top_diagnosis(self) -> str | None:
        """Return the highest-confidence diagnosis, or None."""
        if not self.differential:
            return None
        return max(self.differential, key=lambda x: x.get("confidence", 0.0))["diagnosis"]

    def runner_up(self) -> dict | None:
        """Return the second-highest-confidence entry (runner-up wrong dx target)."""
        if not self.differential or len(self.differential) < 2:
            return None
        sorted_diff = sorted(self.differential, key=lambda x: x.get("confidence", 0.0), reverse=True)
        return sorted_diff[1]


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM output text."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Code fence
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
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_step(text: str) -> ParsedStep:
    """Parse and validate an LLM response against the bias-resilience schema."""
    raw = _extract_json(text)
    if raw is None:
        return ParsedStep(parse_error="could not extract JSON from response")
    if not isinstance(raw, dict):
        return ParsedStep(parse_error=f"expected JSON object, got {type(raw).__name__}")

    errors: list[str] = []

    evidence_summary = raw.get("evidence_summary")
    if not isinstance(evidence_summary, str) or not evidence_summary.strip():
        errors.append("missing or empty 'evidence_summary'")
        evidence_summary = None

    working_diagnosis = raw.get("working_diagnosis")
    if not isinstance(working_diagnosis, str) or not working_diagnosis.strip():
        errors.append("missing or empty 'working_diagnosis'")
        working_diagnosis = None

    differential = raw.get("differential")
    if isinstance(differential, list):
        valid = []
        for i, entry in enumerate(differential):
            if not isinstance(entry, dict):
                errors.append(f"differential[{i}] is not an object")
                continue
            if "diagnosis" not in entry or "confidence" not in entry:
                errors.append(f"differential[{i}] missing 'diagnosis' or 'confidence'")
                continue
            try:
                entry["confidence"] = float(entry["confidence"])
            except (TypeError, ValueError):
                errors.append(f"differential[{i}].confidence not numeric")
                continue
            valid.append({"diagnosis": str(entry["diagnosis"]), "confidence": entry["confidence"]})
        differential = valid if valid else None
        if not differential:
            errors.append("no valid differential entries")
    else:
        errors.append("missing or invalid 'differential'")
        differential = None

    return ParsedStep(
        evidence_summary=evidence_summary,
        working_diagnosis=working_diagnosis,
        differential=differential,
        parse_error="; ".join(errors) if errors else None,
        raw_json=raw,
    )
