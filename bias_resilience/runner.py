"""Per-patient orchestration: run, output paths, cost estimate, idempotency."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import RESULTS_ROOT, ModelConfig, SCHEMA_VERSION, PROMPT_VERSION
from .llm_client import LLMClient
from .replay import get_chunk_elapsed_hours

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"

def _system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text()


def _params_slug(condition_params: dict) -> str:
    if not condition_params:
        return "default"
    parts = []
    for k, v in sorted(condition_params.items()):
        parts.append(f"{k}_{str(v).replace(' ', '_').replace('.', '_')}")
    return "__".join(parts)


def output_path(
    results_root: Path,
    run_id: str,
    model_name: str,
    condition: str,
    anchor: str | None,
    params_slug: str,
    patient_id: str | int,
) -> Path:
    anchor_seg = anchor if anchor else "no_anchor"
    return (
        results_root / "runs" / run_id / model_name / condition
        / anchor_seg / params_slug / f"{patient_id}.json"
    )


def _compute_cost_usd(steps: list[dict], model_cfg: ModelConfig) -> float:
    total_in = sum(s.get("tokens_in", 0) for s in steps)
    total_out = sum(s.get("tokens_out", 0) for s in steps)
    total_think = sum(s.get("tokens_thinking", 0) for s in steps)
    cost = (
        total_in / 1_000_000 * model_cfg.input_price_per_1m
        + total_out / 1_000_000 * model_cfg.output_price_per_1m
        + total_think / 1_000_000 * model_cfg.thinking_price_per_1m
    )
    return round(cost, 6)


def run_patient(
    *,
    timeline_path: Path,
    patient_id: str | int,
    run_id: str,
    model_cfg: ModelConfig,
    condition: str,
    condition_params: dict | None = None,
    anchor: str | None,
    anchor_step: int | None = None,
    target_dx: str | None = None,
    correct_dx: str | None = None,
    original_diff: list[dict] | None = None,
    pathology: str | None = None,
    results_root: Path = RESULTS_ROOT,
    force: bool = False,
    max_steps: int | None = None,
    max_hours: float | None = None,
    chunker_kwargs: dict | None = None,
) -> dict:
    """Run one patient for one condition×anchor and write the output JSON.

    Returns the run record dict (without steps). Skips if output already exists
    unless force=True.
    """
    cparams = condition_params or {}
    slug = _params_slug(cparams)
    out_path = output_path(results_root, run_id, model_cfg.name, condition, anchor, slug, patient_id)

    if out_path.exists() and not force:
        with open(out_path) as f:
            rec = json.load(f)
        return {k: v for k, v in rec.items() if k != "steps"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = LLMClient(model_cfg)
    system = _system_prompt()

    started_at = datetime.now(timezone.utc).isoformat()
    steps: list[dict] = []
    status = "ok"
    status_reason = None
    injection_payload = None

    # anchor_not_resolved: condition requires an anchor but none was found in this timeline.
    if condition != "baseline" and anchor_step is None:
        status = "anchor_not_resolved"
        anchor_label = anchor if anchor else "unknown"
        status_reason = (
            f"anchor '{anchor_label}' could not be identified in this patient's timeline"
        )

    if (
        status == "ok"
        and condition != "baseline"
        and anchor_step is not None
        and max_hours is not None
    ):
        anchor_elapsed = get_chunk_elapsed_hours(timeline_path, anchor_step, chunker_kwargs)
        if anchor_elapsed is not None and anchor_elapsed > max_hours:
            status = "anchor_past_cap"
            status_reason = (
                f"anchor step {anchor_step} is at {anchor_elapsed:.1f}h, "
                f"exceeds {max_hours}h cap"
            )

    if status == "ok":
        try:
            if condition == "baseline":
                from .conditions.baseline import run as run_baseline
                steps = run_baseline(
                    timeline_path, client, system,
                    max_steps=max_steps,
                    max_hours=max_hours,
                    chunker_kwargs=chunker_kwargs,
                )

            elif condition == "struc_belief":
                if anchor_step is None or target_dx is None:
                    raise ValueError("struc_belief requires anchor_step and target_dx")
                if pathology is None:
                    raise ValueError("struc_belief requires pathology")
                from .conditions.struc_belief import run as run_sb, _INJECTED_CONFIDENCE, _DEFAULT_WD_KIND
                injected_conf = float(cparams.get("confidence", _INJECTED_CONFIDENCE))
                wd_kind = str(cparams.get("wd_template", _DEFAULT_WD_KIND))
                steps, injection_payload = run_sb(
                    timeline_path, client, system,
                    anchor_step=anchor_step,
                    target_dx=target_dx,
                    pathology=pathology,
                    correct_dx=correct_dx,
                    original_diff=original_diff,
                    injected_confidence=injected_conf,
                    wd_template_kind=wd_kind,
                    max_steps=max_steps,
                    max_hours=max_hours,
                    chunker_kwargs=chunker_kwargs,
                )

            elif condition in ("pushback_naive", "pushback_counter"):
                if anchor_step is None:
                    raise ValueError(f"{condition} requires anchor_step")
                if condition == "pushback_counter" and target_dx is None:
                    raise ValueError("pushback_counter requires target_dx")
                variant = "naive" if condition == "pushback_naive" else "counter"
                from .conditions.pushback import run as run_pb
                steps, injection_payload = run_pb(
                    timeline_path, client, system,
                    variant=variant,
                    anchor_step=anchor_step,
                    target_dx=target_dx,  # None for naive
                    max_steps=max_steps,
                    max_hours=max_hours,
                    chunker_kwargs=chunker_kwargs,
                )

            elif condition == "struc_consult":
                if anchor_step is None or target_dx is None or original_diff is None:
                    raise ValueError("struc_consult requires anchor_step, target_dx, and original_diff")
                if pathology is None:
                    raise ValueError("struc_consult requires pathology")
                from .conditions.struc_consult import run as run_sc
                steps, injection_payload = run_sc(
                    timeline_path, client, system,
                    anchor_step=anchor_step,
                    target_dx=target_dx,
                    pathology=pathology,
                    original_diff=original_diff,
                    max_steps=max_steps,
                    max_hours=max_hours,
                    chunker_kwargs=chunker_kwargs,
                )

            else:
                raise NotImplementedError(f"Condition '{condition}' not yet implemented")

        except Exception as e:
            status = "failed"
            status_reason = str(e)

    finished_at = datetime.now(timezone.utc).isoformat()

    terminal_dx = None
    if steps:
        last_parsed = steps[-1].get("response_parsed") or {}
        diff = last_parsed.get("differential") or []
        if diff:
            terminal_dx = diff[0].get("diagnosis")

    cost = _compute_cost_usd(steps, model_cfg)

    record = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "patient_id": str(patient_id),
        "model": model_cfg.name,
        "condition": condition,
        "condition_params": cparams,
        "anchor": anchor,
        "status": status,
        "status_reason": status_reason,
        "target_dx": target_dx,
        "injection_step": anchor_step,
        "max_hours": max_hours,
        "injection_payload": injection_payload,
        "started_at": started_at,
        "finished_at": finished_at,
        "terminal_dx": terminal_dx,
        "terminal_correct": None,
        "cost_usd_total": cost,
        "steps": steps,
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    index_path = results_root / "index.jsonl"
    summary = {k: v for k, v in record.items() if k != "steps"}
    with open(index_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    return summary
