"""Utilities for cleaning and matching free-text discharge diagnoses."""

from __future__ import annotations

import re

# Lines that are section headers / separators, not actual diagnoses
_HEADER_RE = re.compile(
    r"^("
    r"primary\s*:?\s*$"
    r"|secondary\s*:?\s*$"
    r"|active\s*:?\s*$"
    r"|acute\s*(issues)?\s*:?\s*$"
    r"|primary\s+issues\s*:?\s*$"
    r"|discharge\s*diagnos[ei]s\s*:?\s*$"
    r"|={3,}"
    r"|_{3,}:?\s*$"
    r")",
    re.IGNORECASE,
)

# Leading prefixes to strip from content lines
_PREFIX_RE = re.compile(
    r"^("
    r"primary\s*diagnos[ei]s?\s*:?\s*"
    r"|#\s*"
    r"|\d+[.)]\s*"
    r"|- "
    r")",
    re.IGNORECASE,
)


def clean_freetextdx(raw: str) -> str:
    """Extract the first meaningful diagnosis line from a DISCHARGE_FREETEXTDX field.

    Skips header/separator lines like "Primary:", "====", "___:" and strips
    common prefixes like "# ", "1. ", "- ".

    Returns the first non-empty content line, or "" if nothing found.
    """
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _HEADER_RE.match(line):
            continue
        cleaned = _PREFIX_RE.sub("", line).strip()
        if cleaned:
            return cleaned
    return ""
