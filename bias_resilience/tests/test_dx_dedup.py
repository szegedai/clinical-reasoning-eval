"""Smoke tests for dx_dedup.is_target_near_twin.

Run with: .venv/bin/python -m bias_resilience.tests.test_dx_dedup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bias_resilience.dx_dedup import is_target_near_twin


CASES = [
    # (candidate, target, expected, label)
    ("Choledocholithiasis", "Choledocholithiasis/Cholangitis", True,
        "compound_first_half"),
    ("Acute Cholangitis", "Choledocholithiasis/Cholangitis", True,
        "compound_second_half_with_modifier"),
    ("Cholangitis NOS", "Choledocholithiasis/Cholangitis", True,
        "compound_second_half_with_qualifier"),
    ("Pancreatitis", "Choledocholithiasis/Cholangitis", False,
        "unrelated_dx"),
    ("Cholecystitis", "Choledocholithiasis/Cholangitis", False,
        "different_biliary_dx_no_token_overlap"),

    ("Pancreatitis", "Pancreatitis", True, "exact_match"),
    ("Acute Pancreatitis", "Pancreatitis", True, "stopword_modifier"),
    ("Recurrent Pancreatitis", "Pancreatitis", True, "stopword_modifier_2"),

    ("Gastritis", "Recurrent Peptic Ulcer Disease (PUD) / Gastritis", True,
        "compound_with_parens_and_acronym"),
    ("Peptic Ulcer Disease", "Recurrent Peptic Ulcer Disease (PUD) / Gastritis", True,
        "two_token_overlap"),
    ("Bowel Obstruction", "Recurrent Peptic Ulcer Disease (PUD) / Gastritis", False,
        "no_overlap_with_compound"),

    ("", "Pancreatitis", False, "empty_candidate"),
    ("Pancreatitis", "", False, "empty_target"),
]


def main() -> int:
    failed = 0
    for candidate, target, expected, label in CASES:
        got = is_target_near_twin(candidate, target)
        ok = got == expected
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}: ({candidate!r}, {target!r}) -> {got}, expected {expected}")
        if not ok:
            failed += 1
    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) FAILED.'}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
