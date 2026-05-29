"""Smoke tests for the 48-hour replay duration cap.

No LLM calls — uses a stub LLMClient that returns a canned ParsedStep JSON.
Run with:
    .venv/bin/python -m bias_resilience.tests.test_replay_cap
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bias_resilience.replay import run_replay
from bias_resilience.llm_client import LLMClient, LLMResponse
from bias_resilience.config import MODELS


# ── Stub LLM client ──────────────────────────────────────────────────────────

_CANNED_RESPONSE = json.dumps({
    "evidence_summary": "test",
    "working_diagnosis": "Appendicitis",
    "differential": [
        {"diagnosis": "Appendicitis", "confidence": 0.80},
        {"diagnosis": "Other", "confidence": 0.20},
    ],
})


class _StubClient:
    """Returns a canned response without any network calls."""
    model_name = "stub"

    def call(self, messages):
        return LLMResponse(
            text=_CANNED_RESPONSE,
            tokens_in=100,
            tokens_out=50,
            tokens_thinking=0,
            latency_ms=1.0,
        )


# ── Synthetic timeline builder ────────────────────────────────────────────────

_BASE_DT = datetime(2100, 1, 1, 0, 0, 0)


def _make_timeline(tmp_dir: Path, elapsed_hours: list[float]) -> Path:
    """Write a minimal timeline CSV matching the real schema."""
    rows = []
    for i, h in enumerate(elapsed_hours):
        event_dt = _BASE_DT + timedelta(hours=h)
        rows.append({
            "subject_id": 99999,
            "hadm_id": 99999,
            "event_time": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_hours": h,
            "time_precision": "exact",
            "source": "ED",
            "event_type": "LAB",
            "description": f"event_{i}",
            "value": str(i),
            "unit": "",
            "flag": "",
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_dir / "timeline_99999.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


_SYSTEM = "You are a diagnostic assistant."
_CHUNKER_KW = {
    "exclude_sources": set(),
    "exclude_event_types": set(),
}


def test_cap_48h_drops_late_chunks():
    """Chunks with min elapsed_hours > 48 are dropped."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 5 events: 0h, 12h, 24h, 48h (at cap), 72h (over cap)
        tp = _make_timeline(tmp, [0.0, 12.0, 24.0, 48.0, 72.0])
        client = _StubClient()
        steps_capped = run_replay(tp, client, _SYSTEM, max_hours=48.0, chunker_kwargs=_CHUNKER_KW)
        steps_uncapped = run_replay(tp, client, _SYSTEM, max_hours=None, chunker_kwargs=_CHUNKER_KW)
        # Cap must reduce or equal chunk count — if 72h is its own chunk, it gets dropped
        assert len(steps_capped) <= len(steps_uncapped), (
            f"cap should not add steps: {len(steps_capped)} > {len(steps_uncapped)}"
        )
        # No events_chunk text should contain the 72h event's description
        for s in steps_capped:
            assert "event_4" not in s.get("events_chunk", ""), \
                "72h event (event_4) should have been dropped by the cap"
        print(f"  PASS [cap_48h_drops_late_chunks] — {len(steps_capped)} capped vs {len(steps_uncapped)} uncapped")


def test_cap_none_includes_all():
    """max_hours=None means no cap — all chunks are included."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [0.0, 24.0, 72.0, 120.0])
        client = _StubClient()
        steps_capped = run_replay(tp, client, _SYSTEM, max_hours=None, chunker_kwargs=_CHUNKER_KW)
        steps_uncapped = run_replay(tp, client, _SYSTEM, chunker_kwargs=_CHUNKER_KW)
        assert len(steps_capped) == len(steps_uncapped), (
            f"cap=None should match uncapped: {len(steps_capped)} vs {len(steps_uncapped)}"
        )
        print(f"  PASS [cap_none_includes_all] — {len(steps_capped)} steps")


def test_all_chunks_within_cap():
    """When all chunks are ≤ cap, nothing is dropped."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp = _make_timeline(tmp, [0.0, 10.0, 20.0, 30.0])
        client = _StubClient()
        steps_capped = run_replay(tp, client, _SYSTEM, max_hours=48.0, chunker_kwargs=_CHUNKER_KW)
        steps_uncapped = run_replay(tp, client, _SYSTEM, max_hours=None, chunker_kwargs=_CHUNKER_KW)
        assert len(steps_capped) == len(steps_uncapped), (
            f"expected same count: {len(steps_capped)} vs {len(steps_uncapped)}"
        )
        print(f"  PASS [all_chunks_within_cap] — {len(steps_capped)} steps")


if __name__ == "__main__":
    tests = [
        test_cap_48h_drops_late_chunks,
        test_cap_none_includes_all,
        test_all_chunks_within_cap,
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
