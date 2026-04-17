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
        system_prompt_onepass: str = "system_prompt_onepass.md",
        step_prompt: str = "step_prompt.md",
        step_prompt_cumulative: str = "step_prompt_cumulative.md",
        onepass_prompt: str = "onepass_prompt.md",
        step_prompt_compressed: str = "step_prompt_compressed.md",
        step_prompt_compressed_initial: str = "step_prompt_compressed_initial.md",
        output_schema: str = "output_schema.md",
        prompts_dir: str | Path | None = None,
    ):
        self._dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
        self._system_prompt = system_prompt
        self._system_prompt_onepass = system_prompt_onepass
        self._step_prompt = step_prompt
        self._step_prompt_cumulative = step_prompt_cumulative
        self._onepass_prompt = onepass_prompt
        self._step_prompt_compressed = step_prompt_compressed
        self._step_prompt_compressed_initial = step_prompt_compressed_initial
        self._output_schema = output_schema
        self._output_schema_onepass = output_schema.replace(".md", "_onepass.md")
        self._output_schema_compressed = output_schema.replace(".md", "_compressed.md")
        self._cache: dict[str, str] = {}
        self._formatter = PromptFormatter()

    def _read(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = (self._dir / name).read_text()
        return self._cache[name]

    def render_system(self, *, onepass: bool = False, compressed: bool = False) -> str:
        name = self._system_prompt_onepass if onepass else self._system_prompt
        template = self._read(name)
        if onepass:
            schema_name = self._output_schema_onepass
        elif compressed:
            schema_name = self._output_schema_compressed
        else:
            schema_name = self._output_schema
        schema = self._read(schema_name)
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

    def render_step_cumulative(
        self,
        chunk,  # ReplayChunk
        global_index: int,
    ) -> str:
        """Render a step prompt with ALL cumulative events (for progressive mode)."""
        template = self._read(self._step_prompt_cumulative)
        cumulative = chunk.cumulative
        all_numbered = self._formatter.format_events_numbered(cumulative, start_index=0)

        new_start_index = global_index
        last_event_index = global_index + len(chunk.events) - 1
        t_end = cumulative["elapsed_hours"].max()

        return _fill(template, {
            "step": str(chunk.step),
            "label": chunk.label,
            "elapsed_hours_end": f"{t_end:.1f}" if pd.notna(t_end) else "?",
            "n_total_events": str(len(cumulative)),
            "n_new_events": str(len(chunk.events)),
            "new_start_index": str(new_start_index),
            "last_event_index": str(last_event_index),
            "formatted_events": "\n".join(all_numbered),
        })

    def render_step_compressed(
        self,
        chunk,  # ReplayChunk
        global_index: int,
        prev_parsed=None,  # ParsedResponse | None
    ) -> str:
        """Render a step prompt with compressed prior reasoning + new events.

        For step 1 (prev_parsed is None), uses the initial template (same as
        conversational mode). For subsequent steps, includes a structured summary
        of the previous assessment plus only the new events.
        """
        events = chunk.events
        numbered = self._formatter.format_events_numbered(events, start_index=global_index)
        last_event_index = global_index + len(events) - 1

        t_start = events["elapsed_hours"].min()
        t_end = events["elapsed_hours"].max()

        base = {
            "step": str(chunk.step),
            "label": chunk.label,
            "elapsed_hours_start": f"{t_start:.1f}" if pd.notna(t_start) else "?",
            "elapsed_hours_end": f"{t_end:.1f}" if pd.notna(t_end) else "?",
            "n_new_events": str(len(events)),
            "formatted_events": "\n".join(numbered),
        }

        if prev_parsed is None or prev_parsed.assessment is None:
            template = self._read(self._step_prompt_compressed_initial)
            return _fill(template, base)

        template = self._read(self._step_prompt_compressed)
        return _fill(template, {
            **base,
            "total_events_so_far": str(global_index + len(events)),
            "first_event_index": str(global_index),
            "last_event_index": str(last_event_index),
            "prev_step": str(chunk.step - 1),
            "prev_assessment": prev_parsed.assessment,
        })

    def render_onepass(self, chunk) -> str:
        """Render a single prompt with all events (for onepass mode)."""
        template = self._read(self._onepass_prompt)
        cumulative = chunk.cumulative
        all_numbered = self._formatter.format_events_numbered(cumulative, start_index=0)

        t_end = cumulative["elapsed_hours"].max()

        return _fill(template, {
            "elapsed_hours_end": f"{t_end:.1f}" if pd.notna(t_end) else "?",
            "n_total_events": str(len(cumulative)),
            "formatted_events": "\n".join(all_numbered),
        })
