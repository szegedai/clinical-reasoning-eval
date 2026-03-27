"""Diagnosis matching using regex patterns from pathologies.yaml.

Two-tier matching:
  - dx_match:    correct diagnosis (counts as accurate)
  - dx_gracious: partial credit (related but not exact)

Usage:
    matcher = DxMatcher()
    matcher.is_correct("acute appendicitis", "appendicitis")       # True
    matcher.is_gracious("RLQ abscess with phlegmon", "appendicitis")  # True
    matcher.match("periappendiceal abscess", "appendicitis")
    # -> ("match", "(?i)appendi.*(abscess|...)")
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class DxMatcher:
    """Regex-based diagnosis matcher loaded from pathologies.yaml."""

    def __init__(self, config_path: Path | None = None):
        path = config_path or (CONFIGS_DIR / "pathologies.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)

        self._match: dict[str, list[re.Pattern]] = {}
        self._gracious: dict[str, list[re.Pattern]] = {}

        for name, pcfg in cfg.items():
            self._match[name] = [
                re.compile(p) for p in pcfg.get("dx_match", [])
            ]
            self._gracious[name] = [
                re.compile(p) for p in pcfg.get("dx_gracious", [])
            ]

    def is_correct(self, diagnosis: str, pathology: str) -> bool:
        """Check if diagnosis matches any dx_match pattern."""
        for pat in self._match.get(pathology, []):
            if pat.search(diagnosis):
                return True
        return False

    def is_gracious(self, diagnosis: str, pathology: str) -> bool:
        """Check if diagnosis matches any dx_gracious pattern (but not dx_match)."""
        for pat in self._gracious.get(pathology, []):
            if pat.search(diagnosis):
                return True
        return False

    def match(self, diagnosis: str, pathology: str) -> tuple[str, str | None]:
        """Return match tier and the pattern that matched.

        Returns:
            ("match", pattern_str)    — correct diagnosis
            ("gracious", pattern_str) — partial credit
            ("none", None)           — no match
        """
        for pat in self._match.get(pathology, []):
            if pat.search(diagnosis):
                return ("match", pat.pattern)
        for pat in self._gracious.get(pathology, []):
            if pat.search(diagnosis):
                return ("gracious", pat.pattern)
        return ("none", None)

    @property
    def pathologies(self) -> list[str]:
        return list(self._match.keys())
