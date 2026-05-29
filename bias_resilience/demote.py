"""Runner-up (wrong-dx) target extraction from baseline replay outputs.

The per-patient bias target is the model's own highest-confidence wrong
diagnosis at the injection anchor step — wrong meaning not correct and not
gracious-correct per DxMatcher. Cached per (patient, model, anchor, pathology).
"""
from __future__ import annotations

import json
from pathlib import Path

from analysis.dx_matcher import DxMatcher

_matcher = DxMatcher()


def _demote_cache_path(
    results_root: Path,
    patient_id: str,
    model_name: str,
    anchor: str,
    pathology: str,
) -> Path:
    return (
        results_root / "cache" / "demote"
        / f"{patient_id}__{model_name}__{anchor}__{pathology}.json"
    )


def extract_runner_up(
    baseline_run: dict,
    anchor_step: int,
    pathology: str,
) -> dict | None:
    steps = baseline_run.get("steps", [])
    for step_rec in steps:
        if step_rec.get("step") != anchor_step:
            continue
        parsed_dict = step_rec.get("response_parsed") or {}
        diff = parsed_dict.get("differential") or []
        if not diff:
            return None

        sorted_diff = sorted(diff, key=lambda x: x.get("confidence", 0.0), reverse=True)
        for entry in sorted_diff:
            dx = entry.get("diagnosis", "")
            if _matcher.is_correct(dx, pathology) or _matcher.is_gracious(dx, pathology):
                continue
            return {"diagnosis": dx, "confidence": entry.get("confidence", 0.0)}

        return None

    return None


def get_or_compute_runner_up(
    *,
    baseline_run: dict,
    anchor_step: int,
    patient_id: str,
    model_name: str,
    anchor: str,
    pathology: str,
    results_root: Path,
    force: bool = False,
) -> dict | None:
    cache_path = _demote_cache_path(results_root, patient_id, model_name, anchor, pathology)

    if cache_path.exists() and not force:
        with open(cache_path) as f:
            return json.load(f)

    runner_up = extract_runner_up(baseline_run, anchor_step, pathology)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(runner_up, f)

    return runner_up


def load_baseline_run(run_path: Path) -> dict:
    with open(run_path) as f:
        return json.load(f)
