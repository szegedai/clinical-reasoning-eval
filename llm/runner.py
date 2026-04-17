"""PatientRunner: run a full temporal replay for one patient against an LLM."""

from __future__ import annotations

import json
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
    first_confident_step: int | None = None

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
            "first_confident_step": self.first_confident_step,
            "steps": [s.to_dict() for s in self.steps],
        }


class PatientRunner:
    """Run a full temporal replay for one patient."""

    VALID_MODES = ("conversational", "progressive", "onepass", "compressed")

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
        stop_after_confidence: int | None = None,
        mode: str = "conversational",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Must be one of {self.VALID_MODES}")
        self.client = client
        self.model = model
        self.renderer = renderer
        self.chunker_kwargs = chunker_kwargs or {}
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_steps = max_steps
        self.stop_after_confidence = stop_after_confidence
        self.mode = mode

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
        """Run full replay for one patient (conversational or progressive mode)."""
        started_at = datetime.now(timezone.utc).isoformat()
        folder = timeline_path.parent
        filename = timeline_path.name

        chunker = TimelineChunker(
            str(folder), filename, **self.chunker_kwargs
        )

        system_prompt = self.renderer.render_system(compressed=self.mode == "compressed")
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        steps: list[StepResult] = []
        total_input = 0
        total_output = 0
        global_index = 0
        steps_since_confidence: int | None = None  # None = not yet confident
        first_confident_step: int | None = None
        prev_parsed: ParsedResponse | None = None  # for compressed mode

        for chunk in chunker.replay():
            # Hard cap: stop after max_steps
            if self.max_steps is not None and chunk.step > self.max_steps:
                break

            # Confidence-based stop: ran enough extra steps after first confidence
            if (self.stop_after_confidence is not None
                    and steps_since_confidence is not None
                    and steps_since_confidence > self.stop_after_confidence):
                break

            if self.mode == "progressive":
                step_prompt = self.renderer.render_step_cumulative(
                    chunk, global_index=global_index,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": step_prompt},
                ]
            elif self.mode == "compressed":
                step_prompt = self.renderer.render_step_compressed(
                    chunk, global_index=global_index,
                    prev_parsed=prev_parsed,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": step_prompt},
                ]
            else:
                step_prompt = self.renderer.render_step(
                    chunk, global_index=global_index,
                )
                messages.append({"role": "user", "content": step_prompt})

            raw_response, input_tokens, output_tokens, latency_ms = self._call_llm(
                messages
            )
            parsed = parse_and_validate(raw_response)

            if self.mode not in ("progressive", "compressed"):
                messages.append({"role": "assistant", "content": raw_response})

            if self.mode == "compressed":
                prev_parsed = parsed

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

            # Track steps since first confidence (never resets)
            if steps_since_confidence is not None:
                steps_since_confidence += 1
            elif parsed.confident_in_diagnosis is True:
                steps_since_confidence = 0
                first_confident_step = chunk.step

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
            first_confident_step=first_confident_step,
        )

    def run_onepass(self, timeline_path: Path, reference_result_path: Path) -> PatientResult:
        """Run a single-shot evaluation using the same events a reference run saw.

        Parameters
        ----------
        timeline_path : Path
            Path to the timeline CSV.
        reference_result_path : Path
            Path to a patient_*.json from a previous run. The termination step
            from that run determines how much of the timeline to include.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        folder = timeline_path.parent
        filename = timeline_path.name

        # Load reference to find termination step
        with open(reference_result_path) as f:
            ref = json.load(f)
        ref_confident = ref.get("first_confident_step")
        ref_steps = ref.get("steps", [])
        if ref_confident is not None:
            target_step = ref_confident
        elif ref_steps:
            target_step = ref_steps[-1]["step"]
        else:
            raise ValueError(f"Reference result has no steps: {reference_result_path}")

        chunker = TimelineChunker(
            str(folder), filename, **self.chunker_kwargs
        )

        system_prompt = self.renderer.render_system(onepass=True)

        # Iterate chunks up to the target step
        last_chunk = None
        global_index = 0
        for chunk in chunker.replay():
            last_chunk = chunk
            global_index += len(chunk.events)
            if chunk.step >= target_step:
                break

        if last_chunk is None:
            raise ValueError(f"No chunks produced for {timeline_path}")

        step_prompt = self.renderer.render_onepass(last_chunk)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": step_prompt},
        ]

        raw_response, input_tokens, output_tokens, latency_ms = self._call_llm(messages)
        parsed = parse_and_validate(raw_response)

        finished_at = datetime.now(timezone.utc).isoformat()

        return PatientResult(
            hadm_id=chunker.hadm_id,
            subject_id=chunker.subject_id,
            model=self.model,
            total_events=chunker.total_events,
            steps=[StepResult(
                step=last_chunk.step,
                label="onepass",
                n_events=len(last_chunk.cumulative),
                prompt=step_prompt,
                raw_response=raw_response,
                parsed=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )],
            system_prompt=system_prompt,
            started_at=started_at,
            finished_at=finished_at,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            first_confident_step=None,
        )
