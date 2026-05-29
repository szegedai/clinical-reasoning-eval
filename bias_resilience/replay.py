"""Core replay loop for bias-resilience experiments.

Each step is a fresh two-message turn; the prior step's JSON is embedded
in the user message alongside the new events. Injection hooks allow
conditions to modify the prior JSON or events at a specific anchor step.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from temporal_replay.chunker import TimelineChunker
from temporal_replay.formatter import PromptFormatter
from .llm_client import LLMClient, LLMResponse
from .schema import ParsedStep, parse_step

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_formatter = PromptFormatter()


def _read_template(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _fill(template: str, replacements: dict[str, str]) -> str:
    for key, val in replacements.items():
        template = template.replace("{" + key + "}", str(val))
    return template


def _time_range_str(events: pd.DataFrame) -> str:
    t_start = events["elapsed_hours"].min()
    t_end = events["elapsed_hours"].max()
    if pd.notna(t_start) and pd.notna(t_end):
        return f", {t_start:.1f}h–{t_end:.1f}h"
    return ""


def _render_initial(step: int, label: str, events: pd.DataFrame, global_idx: int) -> str:
    template = _read_template("step_initial.md")
    numbered = _formatter.format_events_numbered(events, start_index=global_idx)
    return _fill(template, {
        "step": str(step),
        "label": label,
        "n_events": str(len(events)),
        "time_range": _time_range_str(events),
        "formatted_events": "\n".join(numbered),
    })


def _render_followup(
    step: int,
    label: str,
    events: pd.DataFrame,
    global_idx: int,
    prev_step: int,
    prior_parsed: ParsedStep,
) -> str:
    template = _read_template("step_followup.md")
    numbered = _formatter.format_events_numbered(events, start_index=global_idx)
    prior_dict = prior_parsed.to_dict()
    # Remove parse_error before embedding in prompt
    prior_dict.pop("parse_error", None)
    return _fill(template, {
        "prev_step": str(prev_step),
        "prior_json": json.dumps(prior_dict, indent=2),
        "step": str(step),
        "label": label,
        "n_events": str(len(events)),
        "time_range": _time_range_str(events),
        "formatted_events": "\n".join(numbered),
    })


def run_replay(
    timeline_path: Path,
    client: LLMClient,
    system_prompt: str,
    *,
    max_steps: int | None = None,
    max_hours: float | None = None,
    chunker_kwargs: dict | None = None,
    prior_json_transform: Callable[[int, ParsedStep, pd.DataFrame], ParsedStep] | None = None,
    inject_events: Callable[[int, pd.DataFrame], pd.DataFrame] | None = None,
) -> list[dict]:
    ckw = dict(chunker_kwargs) if chunker_kwargs else {}
    chunker = TimelineChunker(str(timeline_path.parent), timeline_path.name, **ckw)

    steps_out: list[dict] = []
    prev_parsed: ParsedStep | None = None
    prev_step_num: int = 0
    global_idx: int = 0

    for chunk in chunker.replay():
        if max_hours is not None and chunk.events["elapsed_hours"].min() > max_hours:
            break
        if max_steps is not None and chunk.step > max_steps:
            break

        events = chunk.events.reset_index(drop=True)

        if inject_events is not None:
            events = inject_events(chunk.step, events)

        effective_prior = prev_parsed
        if prior_json_transform is not None and prev_parsed is not None:
            effective_prior = prior_json_transform(chunk.step, prev_parsed, events)

        if prev_parsed is None:
            prompt = _render_initial(chunk.step, chunk.label, events, global_idx)
        else:
            prompt = _render_followup(
                chunk.step, chunk.label, events, global_idx,
                prev_step_num, effective_prior,  # type: ignore[arg-type]
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        timestamp = datetime.now(timezone.utc).isoformat()
        llm_resp: LLMResponse = client.call(messages)
        parsed = parse_step(llm_resp.text)

        numbered = _formatter.format_events_numbered(events, start_index=global_idx)
        events_chunk_text = "\n".join(numbered)

        steps_out.append({
            "step": chunk.step,
            "label": chunk.label,
            "events_chunk": events_chunk_text,
            "prompt": prompt,
            "response_raw": llm_resp.text,
            "response_parsed": parsed.to_dict(),
            "tokens_in": llm_resp.tokens_in,
            "tokens_out": llm_resp.tokens_out,
            "tokens_thinking": llm_resp.tokens_thinking,
            "latency_ms": round(llm_resp.latency_ms, 1),
            "timestamp": timestamp,
        })

        global_idx += len(events)
        prev_parsed = parsed
        prev_step_num = chunk.step

    return steps_out


def get_chunk_elapsed_hours(
    timeline_path: Path,
    step: int,
    chunker_kwargs: dict | None = None,
) -> float | None:
    ckw = dict(chunker_kwargs) if chunker_kwargs else {}
    chunker = TimelineChunker(str(timeline_path.parent), timeline_path.name, **ckw)
    for chunk in chunker.replay():
        if chunk.step == step:
            return float(chunk.events["elapsed_hours"].min())
        if chunk.step > step:
            break
    return None
