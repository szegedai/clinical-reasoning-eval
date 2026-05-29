"""Smoke tests for struc_belief._build_biased_differential.

No LLM calls. Run with:
    .venv/bin/python -m bias_resilience.tests.test_struc_belief
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.dx_matcher import DxMatcher
from bias_resilience.conditions.struc_belief import _build_biased_differential, _INJECTED_CONFIDENCE

matcher = DxMatcher()
PATHOLOGY = "cholecystitis"


def _assert_invariants(result: list[dict], target_dx: str, pathology: str, label: str):
    assert len(result) == 5, f"[{label}] expected 5 entries, got {len(result)}"
    assert result[0]["diagnosis"] == target_dx, \
        f"[{label}] position 0 is '{result[0]['diagnosis']}', expected '{target_dx}'"
    assert result[0]["confidence"] == _INJECTED_CONFIDENCE, \
        f"[{label}] target confidence is {result[0]['confidence']}, expected {_INJECTED_CONFIDENCE}"
    for d in result[1:]:
        assert d["confidence"] < _INJECTED_CONFIDENCE, \
            f"[{label}] '{d['diagnosis']}' conf {d['confidence']} >= target {_INJECTED_CONFIDENCE}"
        assert not matcher.is_correct(d["diagnosis"], pathology) and \
               not matcher.is_gracious(d["diagnosis"], pathology), \
            f"[{label}] correct/gracious entry '{d['diagnosis']}' in final differential"
    total = sum(d["confidence"] for d in result)
    assert abs(total - 1.0) < 1e-3, \
        f"[{label}] confidences sum to {total:.4f}, expected 1.0"
    print(f"  PASS [{label}]")
    for d in result:
        print(f"    {d['confidence']:.4f}  {d['diagnosis']}")


def test_single_correct_dropped():
    """One correct entry dropped, one filler from BIAS_PACKAGES."""
    original = [
        {"diagnosis": "Cholecystitis", "confidence": 0.50},   # correct → removed
        {"diagnosis": "Pancreatitis", "confidence": 0.20},
        {"diagnosis": "Gastritis", "confidence": 0.15},
        {"diagnosis": "GERD", "confidence": 0.10},
        {"diagnosis": "Peptic ulcer", "confidence": 0.05},
    ]
    result, removed, fillers = _build_biased_differential(
        original, "Pancreatitis", PATHOLOGY, matcher
    )
    assert "Cholecystitis" in removed, f"expected Cholecystitis removed, got {removed}"
    assert len(fillers) == 1, f"expected 1 filler, got {fillers}"
    _assert_invariants(result, "Pancreatitis", PATHOLOGY, "single_correct_dropped")


def test_multiple_correct_dropped():
    """Two correct/gracious entries (Cholecystitis + Biliary Colic) dropped, two fillers."""
    original = [
        {"diagnosis": "Biliary Colic", "confidence": 0.40},    # gracious → removed
        {"diagnosis": "Cholecystitis", "confidence": 0.25},    # correct → removed
        {"diagnosis": "Pancreatitis", "confidence": 0.20},
        {"diagnosis": "GERD", "confidence": 0.10},
        {"diagnosis": "Gastritis", "confidence": 0.05},
    ]
    result, removed, fillers = _build_biased_differential(
        original, "Pancreatitis", PATHOLOGY, matcher
    )
    assert len(removed) == 2, f"expected 2 removed, got {removed}"
    assert len(fillers) == 2, f"expected 2 fillers, got {fillers}"
    _assert_invariants(result, "Pancreatitis", PATHOLOGY, "multiple_correct_dropped")


def test_target_already_in_diff():
    """Target is present in original_diff — should not be duplicated."""
    original = [
        {"diagnosis": "Biliary Colic", "confidence": 0.40},
        {"diagnosis": "Pancreatitis", "confidence": 0.30},    # target
        {"diagnosis": "Cholecystitis", "confidence": 0.20},  # correct → removed
        {"diagnosis": "GERD", "confidence": 0.10},
    ]
    result, removed, fillers = _build_biased_differential(
        original, "Pancreatitis", PATHOLOGY, matcher
    )
    assert sum(1 for d in result if d["diagnosis"] == "Pancreatitis") == 1, \
        "target_dx appears more than once"
    _assert_invariants(result, "Pancreatitis", PATHOLOGY, "target_already_in_diff")


def test_empty_original_diff():
    """Empty differential — all 4 others come from BIAS_PACKAGES."""
    result, removed, fillers = _build_biased_differential(
        [], "Pancreatitis", PATHOLOGY, matcher
    )
    assert removed == [], f"expected no removed, got {removed}"
    assert len(fillers) == 4, f"expected 4 fillers, got {fillers}"
    _assert_invariants(result, "Pancreatitis", PATHOLOGY, "empty_original_diff")


def test_bias_packages_exhausted():
    """Pool empty (unknown pathology) — 'Other abdominal pathology' last-resort kicks in."""
    # Using an unrecognised pathology key gives an empty BIAS_PACKAGES pool.
    # With only 1 kept entry from original_diff, 3 last-resort fillers fire.
    original = [
        {"diagnosis": "Something Correct", "confidence": 0.50},  # not correct for unknown_path
        {"diagnosis": "Wrong Dx A", "confidence": 0.30},
    ]
    # For "unknown_pathology" DxMatcher returns is_correct=False for everything,
    # so both entries are kept (2 kept). Pool is empty → 2 last-resort fillers.
    result, removed, fillers = _build_biased_differential(
        original, "Target Dx", "unknown_pathology", matcher
    )
    assert any(d["diagnosis"] == "Other abdominal pathology" for d in result[1:]), \
        f"expected 'Other abdominal pathology' as last-resort: {result}"
    assert len(fillers) >= 2, f"expected last-resort fillers, got {fillers}"
    # Can't use full _assert_invariants for unknown_pathology (no matcher rules),
    # so check the numeric invariants manually.
    assert len(result) == 5
    assert result[0]["diagnosis"] == "Target Dx"
    assert result[0]["confidence"] == _INJECTED_CONFIDENCE
    assert max(d["confidence"] for d in result[1:]) < _INJECTED_CONFIDENCE
    assert abs(sum(d["confidence"] for d in result) - 1.0) < 1e-3
    print(f"  PASS [pool_exhausted_lastresort]")
    for d in result:
        print(f"    {d['confidence']:.4f}  {d['diagnosis']}")


def test_compound_target_dedup():
    """Compound target like 'Choledocholithiasis/Cholangitis' should drop both
    'Choledocholithiasis' and 'Acute Cholangitis' from kept entries (the
    20275759 pattern — exact-match dedup let near-twins survive).
    """
    original = [
        {"diagnosis": "Choledocholithiasis/Cholangitis", "confidence": 0.30},  # target
        {"diagnosis": "Choledocholithiasis", "confidence": 0.25},              # near-twin → dropped
        {"diagnosis": "Acute Cholangitis", "confidence": 0.15},                # near-twin → dropped
        {"diagnosis": "Cholecystitis", "confidence": 0.20},                    # correct → removed
        {"diagnosis": "Pancreatitis", "confidence": 0.10},                     # kept
    ]
    result, removed, fillers = _build_biased_differential(
        original, "Choledocholithiasis/Cholangitis", PATHOLOGY, matcher
    )
    diagnoses = [d["diagnosis"] for d in result[1:]]
    assert "Choledocholithiasis" not in diagnoses, \
        f"near-twin 'Choledocholithiasis' leaked into final differential: {diagnoses}"
    assert "Acute Cholangitis" not in diagnoses, \
        f"near-twin 'Acute Cholangitis' leaked into final differential: {diagnoses}"
    _assert_invariants(result, "Choledocholithiasis/Cholangitis", PATHOLOGY, "compound_target_dedup")


def test_sorted_by_confidence():
    """Others must be monotonically non-increasing in positions 1-4.

    Reproduces the 20075398 pattern: two low-conf kept entries get outranked
    by higher-weight fillers — after the sort, positions 1-4 must be ordered.
    """
    original = [
        {"diagnosis": "Biliary Colic", "confidence": 0.40},   # gracious → removed
        {"diagnosis": "Cholecystitis", "confidence": 0.20},   # correct → removed
        {"diagnosis": "GERD exacerbation", "confidence": 0.08},  # kept (low)
        {"diagnosis": "Peptic ulcer", "confidence": 0.05},       # kept (lower)
        {"diagnosis": "Pancreatitis", "confidence": 0.15},        # target
    ]
    result, removed, fillers = _build_biased_differential(
        original, "Pancreatitis", PATHOLOGY, matcher
    )
    confs = [d["confidence"] for d in result[1:]]
    for i in range(len(confs) - 1):
        assert confs[i] >= confs[i + 1], \
            f"non-monotonic at positions {i+1},{i+2}: {confs[i]} < {confs[i+1]}\nresult={result}"
    _assert_invariants(result, "Pancreatitis", PATHOLOGY, "sorted_by_confidence")


if __name__ == "__main__":
    tests = [
        test_single_correct_dropped,
        test_multiple_correct_dropped,
        test_target_already_in_diff,
        test_empty_original_diff,
        test_bias_packages_exhausted,
        test_compound_target_dedup,
        test_sorted_by_confidence,
    ]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
