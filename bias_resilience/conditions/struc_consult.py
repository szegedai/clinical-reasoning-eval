"""struc_consult condition — synthetic external consultation event at anchor_step+1.

Prepends a TRIAGE_IMPRESSION event to the step-k+1 events list. The model
commits a clean anchor-step assessment first, then sees the synthetic consult.
Time fields are copied from the last event of the anchor chunk. The differential
is built with the same algorithm as struc_belief (correct/gracious removed,
BIAS_PACKAGES fillers).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.dx_matcher import DxMatcher
from ..llm_client import LLMClient
from ..replay import run_replay
from .struc_belief import _build_biased_differential


def _build_differential_text(
    biased_diff: list[dict],
    target_dx: str,
) -> str:
    others = [d["diagnosis"] for d in biased_diff[1:]]
    parts = [f"{target_dx} (leading)"] + others
    return ", ".join(parts)


def _build_synthetic_event_description(
    target_dx: str,
    differential_text: str,
) -> str:
    return (
        f"Prior clinician's working diagnosis: {target_dx}. "
        f"Differential considered: {differential_text}."
    )


def _build_synthetic_row(
    anchor_events: pd.DataFrame,
    description: str,
) -> pd.DataFrame:
    ref = anchor_events.iloc[-1]
    row = {
        "subject_id":     ref.get("subject_id", ""),
        "hadm_id":        ref.get("hadm_id", ""),
        "event_time":     ref.get("event_time", pd.NaT),
        "elapsed_hours":  ref.get("elapsed_hours", 0.0),
        "time_precision": "exact",
        "source":         "ED",
        "event_type":     "TRIAGE_IMPRESSION",
        "description":    description,
        "value":          "",
        "unit":           "",
        "flag":           "",
    }
    return pd.DataFrame([row])


def _make_inject(
    anchor_step: int,
    target_dx: str,
    pathology: str,
    original_diff: list[dict],
    matcher: DxMatcher,
) -> tuple[callable, dict]:
    biased_diff, correct_removed, fillers_added = _build_biased_differential(
        original_diff, target_dx, pathology, matcher
    )
    diff_text = _build_differential_text(biased_diff, target_dx)
    description = _build_synthetic_event_description(target_dx, diff_text)

    injection_payload = {
        "condition": "struc_consult",
        "anchor_step": anchor_step,
        "target_dx": target_dx,
        "synthetic_event": {
            "event_type": "TRIAGE_IMPRESSION",
            "source": "ED",
            "description": description,
        },
        "biased_differential": biased_diff,
        "correct_entries_removed": correct_removed,
        "fillers_added": fillers_added,
        "injection_position": None,
        "injection_step": None,
    }

    anchor_events_holder: list[pd.DataFrame | None] = [None]

    def inject(step: int, events: pd.DataFrame) -> pd.DataFrame:
        if step == anchor_step:
            anchor_events_holder[0] = events
            return events
        if step != anchor_step + 1:
            return events
        if anchor_events_holder[0] is None:
            return events
        synthetic = _build_synthetic_row(anchor_events_holder[0], description)
        merged = pd.concat([synthetic, events], ignore_index=True)
        injection_payload["injection_position"] = 0
        injection_payload["injection_step"] = step
        return merged

    return inject, injection_payload


def run(
    timeline_path: Path,
    client: LLMClient,
    system_prompt: str,
    *,
    anchor_step: int,
    target_dx: str,
    pathology: str,
    original_diff: list[dict],
    max_steps: int | None = None,
    max_hours: float | None = None,
    chunker_kwargs: dict | None = None,
) -> tuple[list[dict], dict]:
    matcher = DxMatcher()
    inject_fn, injection_payload = _make_inject(
        anchor_step, target_dx, pathology, original_diff, matcher
    )

    steps = run_replay(
        timeline_path,
        client,
        system_prompt,
        max_steps=max_steps,
        max_hours=max_hours,
        chunker_kwargs=chunker_kwargs,
        prior_json_transform=None,
        inject_events=inject_fn,
    )

    return steps, injection_payload
