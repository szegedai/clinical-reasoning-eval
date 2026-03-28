"""PatientRunner: run a full temporal replay for one patient against an LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import openai

from temporal_replay import TimelineChunker, PromptRenderer
from .parser import ParsedResponse, parse_and_validate


@dataclass
class StepResult:
    """Result from a single replay step."""

    step: int
    label: str
    n_events: int
    prompt: str
    raw_response: str
    parsed: ParsedResponse | None
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def to_dict(self) -> dict:
        d = {
            "step": self.step,
            "label": self.label,
            "n_events": self.n_events,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }
        if self.parsed:
            d["parsed"] = {
                "assessment": self.parsed.assessment,
                "delta": self.parsed.delta,
                "differential": self.parsed.differential,
                "key_findings": self.parsed.key_findings,
                "actions": self.parsed.actions,
                "confident_in_diagnosis": self.parsed.confident_in_diagnosis,
                "raw_json": self.parsed.raw_json,
                "parse_error": self.parsed.parse_error,
            }
        else:
            d["parsed"] = None
        return d


@dataclass
class PatientResult:
    """Result from a full patient replay."""

    hadm_id: int
    subject_id: int
    model: str
    total_events: int
    steps: list[StepResult]
    system_prompt: str
    started_at: str
    finished_at: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "hadm_id": self.hadm_id,
            "subject_id": self.subject_id,
            "model": self.model,
            "total_events": self.total_events,
            "system_prompt": self.system_prompt,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "steps": [s.to_dict() for s in self.steps],
        }


class PatientRunner:
    """Run a full temporal replay for one patient."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        renderer: PromptRenderer,
        *,
        chunker_kwargs: dict | None = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_steps: int | None = None,
    ):
        self.client = client
        self.model = model
        self.renderer = renderer
        self.chunker_kwargs = chunker_kwargs or {}
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_steps = max_steps

    def _call_llm(self, messages: list[dict]) -> tuple[str, int, int, float]:
        """Call LLM with retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                text = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                return text, input_tokens, output_tokens, latency_ms

            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)

        raise last_error  # type: ignore[misc]

    def run(self, timeline_path: Path) -> PatientResult:
        """Run full replay for one patient."""
        started_at = datetime.now(timezone.utc).isoformat()
        folder = timeline_path.parent
        filename = timeline_path.name

        chunker = TimelineChunker(
            str(folder), filename, **self.chunker_kwargs
        )

        system_prompt = self.renderer.render_system()
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        steps: list[StepResult] = []
        total_input = 0
        total_output = 0
        global_index = 0

        for chunk in chunker.replay():
            # Hard cap: stop after max_steps
            if self.max_steps is not None and chunk.step > self.max_steps:
                break

            step_prompt = self.renderer.render_step(
                chunk, global_index=global_index,
            )
            messages.append({"role": "user", "content": step_prompt})

            raw_response, input_tokens, output_tokens, latency_ms = self._call_llm(
                messages
            )
            parsed = parse_and_validate(raw_response)

            messages.append({"role": "assistant", "content": raw_response})

            total_input += input_tokens
            total_output += output_tokens

            steps.append(StepResult(
                step=chunk.step,
                label=chunk.label,
                n_events=len(chunk.events),
                prompt=step_prompt,
                raw_response=raw_response,
                parsed=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            ))

            global_index += len(chunk.events)

        finished_at = datetime.now(timezone.utc).isoformat()

        return PatientResult(
            hadm_id=chunker.hadm_id,
            subject_id=chunker.subject_id,
            model=self.model,
            total_events=chunker.total_events,
            steps=steps,
            system_prompt=system_prompt,
            started_at=started_at,
            finished_at=finished_at,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        )
