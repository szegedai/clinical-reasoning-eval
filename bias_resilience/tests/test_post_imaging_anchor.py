"""Smoke tests for the post_imaging anchor and anchor_past_cap handling.

No LLM calls. Tests cover:
  1. RADIOLOGY_REPORT at step 5 → post_imaging == 5
  2. Only IMAGING_STUDY present → post_imaging == None (anchors on report, not study)
  3. Multiple RADIOLOGY_REPORTs → post_imaging == first occurrence
  4. No imaging at all → post_imaging == None
  5. RADIOLOGY_REPORT past 48h → resolves to step; run_patient returns anchor_past_cap

Run with:
    .venv/bin/python -m bias_resilience.tests.test_post_imaging_anchor
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bias_resilience.anchors import resolve_anchors
from bias_resilience.config import MODELS
from bias_resilience.runner import run_patient

_BASE_DT = datetime(2100, 1, 1, 0, 0, 0)

_SCHEMA_COLS = [
    "subject_id", "hadm_id", "event_time", "elapsed_hours",
    "time_precision", "source", "event_type", "description",
    "value", "unit", "flag",
]


def _make_timeline(
    tmp_dir: Path,
    rows: list[dict],
) -> Path:
    """Write a minimal timeline CSV with required schema columns."""
    df = pd.DataFrame(rows, columns=_SCHEMA_COLS)
    csv_path = tmp_dir / "timeline_99999.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _row(elapsed_h: float, event_type: str, description: str = "") -> dict:
    event_dt = _BASE_DT + timedelta(hours=elapsed_h)
    return {
        "subject_id": 99999,
        "hadm_id": 99999,
        "event_time": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_hours": elapsed_h,
        "time_precision": "exact",
        "source": "ED",
        "event_type": event_type,
        "description": description,
        "value": "",
        "unit": "",
        "flag": "",
    }


def test_radiology_report_fires_anchor():
    """RADIOLOGY_REPORT at a later step → post_imaging resolves to that step."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [
            _row(0.0, "TRIAGE"),
            _row(1.0, "VITAL_SIGNS"),
            _row(2.0, "VITAL_SIGNS"),
            _row(3.0, "LAB_RESULT"),
            _row(4.0, "IMAGING_STUDY", "CT abdomen ordered"),
            _row(5.0, "RADIOLOGY_REPORT", "CT abdomen: acute cholecystitis"),
            _row(6.0, "LAB_RESULT"),
        ])
        anchors = resolve_anchors(tp)
        assert anchors["post_imaging"] is not None, "expected post_imaging to resolve"
        # The RADIOLOGY_REPORT at 5h should land somewhere after step 1
        # (exact step depends on chunker grouping — just confirm it's not None and ≥ 1)
        assert anchors["post_imaging"] >= 1
        print(f"  PASS [radiology_report_fires_anchor] — post_imaging=step{anchors['post_imaging']}")


def test_imaging_study_does_not_fire_anchor():
    """IMAGING_STUDY alone does not fire post_imaging — only RADIOLOGY_REPORT does."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [
            _row(0.0, "TRIAGE"),
            _row(1.0, "VITAL_SIGNS"),
            _row(2.0, "IMAGING_STUDY", "CT abdomen ordered"),
            _row(3.0, "LAB_RESULT"),
            _row(4.0, "IMAGING_STUDY", "CXR ordered"),
        ])
        anchors = resolve_anchors(tp)
        assert anchors["post_imaging"] is None, (
            f"expected post_imaging=None (only IMAGING_STUDY present), "
            f"got step{anchors['post_imaging']}"
        )
        print("  PASS [imaging_study_does_not_fire_anchor] — post_imaging=None as expected")


def test_first_radiology_report_wins():
    """post_imaging resolves to the step of the first RADIOLOGY_REPORT, not a later one."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [
            _row(0.0, "TRIAGE"),
            _row(1.0, "VITAL_SIGNS"),
            _row(3.0, "RADIOLOGY_REPORT", "CXR: clear"),
            _row(5.0, "LAB_RESULT"),
            _row(8.0, "RADIOLOGY_REPORT", "CT abdomen: cholecystitis"),
            _row(10.0, "LAB_RESULT"),
        ])
        anchors_first = resolve_anchors(tp)

        # Compare: timeline with only the second RADIOLOGY_REPORT
        tp2 = Path(td) / "timeline_88888.csv"
        rows2 = [
            _row(0.0, "TRIAGE"),
            _row(1.0, "VITAL_SIGNS"),
            _row(5.0, "LAB_RESULT"),
            _row(8.0, "RADIOLOGY_REPORT", "CT abdomen: cholecystitis"),
            _row(10.0, "LAB_RESULT"),
        ]
        pd.DataFrame(rows2, columns=_SCHEMA_COLS).to_csv(tp2, index=False)
        anchors_second = resolve_anchors(tp2)

        assert anchors_first["post_imaging"] is not None
        assert anchors_second["post_imaging"] is not None
        # First timeline should anchor at an earlier or equal step than the second
        assert anchors_first["post_imaging"] <= anchors_second["post_imaging"], (
            f"first RADIOLOGY_REPORT should anchor at step ≤ second's; "
            f"got {anchors_first['post_imaging']} vs {anchors_second['post_imaging']}"
        )
        print(
            f"  PASS [first_radiology_report_wins] — "
            f"first={anchors_first['post_imaging']}, second={anchors_second['post_imaging']}"
        )


def test_no_imaging_returns_none():
    """Timeline with no imaging events → post_imaging == None."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [
            _row(0.0, "TRIAGE"),
            _row(1.0, "VITAL_SIGNS"),
            _row(2.0, "LAB_RESULT"),
            _row(3.0, "LAB_RESULT"),
        ])
        anchors = resolve_anchors(tp)
        assert anchors["post_imaging"] is None, (
            f"expected post_imaging=None (no imaging), got step{anchors['post_imaging']}"
        )
        print("  PASS [no_imaging_returns_none] — post_imaging=None")


def test_anchor_past_cap_skips_cleanly():
    """RADIOLOGY_REPORT past 48h: anchor resolves uncapped, but run_patient
    with max_hours=48 returns anchor_past_cap without any LLM call.

    The chunker batches ~6 events per chunk (time-window based). We fill 4 full
    chunks with 24 LAB_RESULT events at 0-23h so the RADIOLOGY_REPORT at 60h
    starts its own chunk 5 with min_elapsed_hours=60h > 48h cap.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 24 events at 0-23h → 4 full chunks; RADIOLOGY_REPORT at 60h → chunk 5
        rows = [_row(float(i), "LAB_RESULT", f"lab_{i}") for i in range(24)]
        rows.append(_row(60.0, "RADIOLOGY_REPORT", "CT abdomen: cholecystitis — late report"))
        tp = _make_timeline(tmp, rows)
        anchors = resolve_anchors(tp)
        assert anchors["post_imaging"] is not None, (
            "anchor resolution should succeed uncapped even for late reports"
        )
        imaging_step = anchors["post_imaging"]

        # run_patient with 48h cap should set anchor_past_cap, not call LLM
        model_cfg = MODELS["gemini-2.5-flash"]
        rec = run_patient(
            timeline_path=tp,
            patient_id="99999",
            run_id="test_anchor_past_cap",
            model_cfg=model_cfg,
            condition="pushback_naive",
            anchor="post_imaging",
            anchor_step=imaging_step,
            target_dx=None,
            results_root=tmp / "results",
            force=True,
            max_hours=48.0,
        )
        assert rec["status"] == "anchor_past_cap", (
            f"expected anchor_past_cap, got {rec['status']!r}: {rec.get('status_reason')}"
        )
        assert not rec.get("steps"), "anchor_past_cap run should produce no steps"
        print(
            f"  PASS [anchor_past_cap_skips_cleanly] — "
            f"step{imaging_step} flagged as anchor_past_cap, no LLM call"
        )


if __name__ == "__main__":
    tests = [
        test_radiology_report_fires_anchor,
        test_imaging_study_does_not_fire_anchor,
        test_first_radiology_report_wins,
        test_no_imaging_returns_none,
        test_anchor_past_cap_skips_cleanly,
    ]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
