"""
Reads prompt templates and fills placeholders for the temporal replay system.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .formatter import PromptFormatter

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _fill(template: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        template = template.replace("{" + key + "}", str(value))
    return template


class PromptRenderer:
    """Render prompt templates with event data.

    Parameters
    ----------
    system_prompt : str
        Filename of the system prompt template (default: "system_prompt.md").
    step_prompt : str
        Filename of the step prompt template (default: "step_prompt.md").
    output_schema : str
        Filename of the output schema (default: "output_schema.md").
    prompts_dir : str or Path, optional
        Directory containing the template files.

    Usage:
        renderer = PromptRenderer()
        renderer = PromptRenderer(system_prompt="system_prompt_v2.md")
        system_msg = renderer.render_system()
        step_msg = renderer.render_step(chunk, global_index=0)
    """

    def __init__(
        self,
        *,
        system_prompt: str = "system_prompt.md",
        step_prompt: str = "step_prompt.md",
        output_schema: str = "output_schema.md",
        prompts_dir: str | Path | None = None,
    ):
        self._dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
        self._system_prompt = system_prompt
        self._step_prompt = step_prompt
        self._output_schema = output_schema
        self._cache: dict[str, str] = {}
        self._formatter = PromptFormatter()

    def _read(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = (self._dir / name).read_text()
        return self._cache[name]

    def render_system(self) -> str:
        template = self._read(self._system_prompt)
        schema = self._read(self._output_schema)
        return _fill(template, {"output_schema": schema})

    def render_step(
        self,
        chunk,  # ReplayChunk
        global_index: int,
    ) -> str:
        """Render a step prompt for a single replay chunk.

        Parameters
        ----------
        chunk : ReplayChunk
            The chunk from TimelineChunker.replay().
        global_index : int
            The global event index of the first event in this chunk.
        """
        template = self._read(self._step_prompt)
        events = chunk.events
        numbered = self._formatter.format_events_numbered(events, start_index=global_index)
        last_event_index = global_index + len(events) - 1

        t_start = events["elapsed_hours"].min()
        t_end = events["elapsed_hours"].max()

        return _fill(template, {
            "step": str(chunk.step),
            "label": chunk.label,
            "elapsed_hours_start": f"{t_start:.1f}" if pd.notna(t_start) else "?",
            "elapsed_hours_end": f"{t_end:.1f}" if pd.notna(t_end) else "?",
            "n_new_events": str(len(events)),
            "formatted_events": "\n".join(numbered),
            "last_event_index": str(last_event_index),
        })
