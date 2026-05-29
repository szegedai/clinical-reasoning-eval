"""Baseline condition: clean replay with no manipulation."""
from __future__ import annotations

from pathlib import Path

from ..llm_client import LLMClient
from ..replay import run_replay


def run(
    timeline_path: Path,
    client: LLMClient,
    system_prompt: str,
    *,
    max_steps: int | None = None,
    max_hours: float | None = None,
    chunker_kwargs: dict | None = None,
) -> list[dict]:
    """Run the baseline (no injection) replay for one patient."""
    return run_replay(
        timeline_path,
        client,
        system_prompt,
        max_steps=max_steps,
        max_hours=max_hours,
        chunker_kwargs=chunker_kwargs,
        prior_json_transform=None,
        inject_events=None,
    )
