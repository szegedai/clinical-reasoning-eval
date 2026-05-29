"""FeatureExtractor: extract clinical features from one patient timeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
import pandas as pd

from temporal_replay import TimelineChunker, PromptFormatter
from llm.parser import parse_llm_json


FORMATTER = PromptFormatter()

SYSTEM_PROMPT = """\
You are a clinical informatics expert. Extract the presence or absence of \
specific clinical features from a patient's emergency department timeline.

For each feature, respond with:
- "value": "yes", "no", or "not_stated"
- "event_index": the [N] index of the timeline event that most directly \
supports your answer, or null if not_stated

Rules for choosing the value:
- "yes": the feature is clearly present in the timeline.
- "no": the feature would reasonably have been mentioned or observed at a \
given point in the timeline, but was not. Anchor to the event where it \
would have appeared. Examples:
  * An HPI describes abdominal pain but does not mention nausea → "no" for \
nausea, anchored to the HPI event (patients are asked about nausea).
  * A physical exam documents the abdomen but does not mention RLQ \
tenderness → "no", anchored to the PE event.
  * A CT report describes the abdomen but does not mention appendiceal \
dilation → "no", anchored to the CT event.
  * Lab results include a CBC but no lipase → "not_stated" for lipase \
(it was not ordered).
- "not_stated": there is no event in the timeline where this feature would \
reasonably have been observed or reported. Example: no CT was done → any \
CT-based feature is "not_stated". No lipase ordered → lipase is "not_stated".

Mapping features to relevant events:
- Symptom features (nausea, pain location, etc.) → HPI or nursing notes
- Exam features (tenderness, guarding, etc.) → physical exam
- Lab features (WBC elevated, lipase, etc.) → lab result events. If the \
specific test was not ordered, use "not_stated".
- Imaging features (appendix dilated, fat stranding, free fluid, etc.) → \
radiology report (CT, US, MRI). A radiology report for the relevant body \
region IS the anchoring event for imaging features. If the report does not \
mention a finding, that is "no", not "not_stated".

Prefer "no" over "not_stated" when a relevant event exists that would have \
captured the feature if it were present. Only use "not_stated" when the \
relevant exam, test, or history was truly never obtained."""

USER_TEMPLATE = """\
PATIENT TIMELINE:
{timeline}

FEATURES TO EXTRACT (with event anchoring):
{feature_list}

For each feature, respond with a JSON object:
- "value": "yes", "no", or "not_stated"
- "event_index": the [N] index supporting your answer, or null if not_stated

"no" = a relevant event exists where this feature would have been \
mentioned if present, but it was not. Anchor to that event.
"not_stated" = no relevant event exists in the timeline at all \
(the exam/test/history was never obtained).

```json
{{
{feature_json_example}
}}
```"""


@dataclass
class FeatureResult:
    """Extraction result for a single feature."""
    value: str           # yes, no, not_stated
    event_index: int | None
    step: int | None


@dataclass
class ChunkBoundary:
    """Event index range for one chunk step."""
    step: int
    label: str
    n_events: int
    first_event_index: int
    max_event_index: int

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "label": self.label,
            "n_events": self.n_events,
            "first_event_index": self.first_event_index,
            "max_event_index": self.max_event_index,
        }


@dataclass
class ExtractionResult:
    """Result of feature extraction for one patient."""
    hadm_id: str
    n_events: int
    n_chunks: int
    boundaries: list[ChunkBoundary]
    features: dict[str, FeatureResult]

    def to_dict(self) -> dict:
        return {
            "hadm_id": self.hadm_id,
            "n_events": self.n_events,
            "n_chunks": self.n_chunks,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "features": {
                name: {"value": f.value, "event_index": f.event_index, "step": f.step}
                for name, f in self.features.items()
            },
        }


class FeatureExtractor:
    """Extract clinical features with event anchoring from a single patient."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        features: list[dict],
        *,
        chunker_kwargs: dict | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.client = client
        self.model = model
        self.features = features
        self.feat_names = [f["name"] for f in features]
        self.chunker_kwargs = chunker_kwargs or {}
        self.temperature = temperature
        self.max_retries = max_retries

        # Pre-build prompt fragments
        self.feat_str = "\n".join(
            f"- {f['name']}: {f['description']}" for f in features)
        self.feat_json = ",\n  ".join(
            f'"{f["name"]}": {{"value": "yes|no|not_stated", "event_index": N|null}}'
            for f in features[:3]
        ) + ",\n  ..."

    def _call_llm(self, system: str, user: str) -> str | None:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content or ""
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError):
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
        return None

    def _get_boundaries(self, timeline_path: Path) -> list[ChunkBoundary]:
        chunker = TimelineChunker(
            timeline_path.parent, timeline_path.name, **self.chunker_kwargs)
        boundaries = []
        cumulative = 0
        for chunk in chunker.replay():
            n = len(chunk.events)
            boundaries.append(ChunkBoundary(
                step=chunk.step,
                label=chunk.label,
                n_events=n,
                first_event_index=cumulative,
                max_event_index=cumulative + n - 1,
            ))
            cumulative += n
        return boundaries

    def _load_filtered(self, timeline_path: Path) -> pd.DataFrame:
        df = pd.read_csv(timeline_path, parse_dates=["event_time"])
        df = df[df["time_precision"] != "date_only"].copy()

        exclude_sources = self.chunker_kwargs.get("exclude_sources")
        if exclude_sources:
            df = df[~df["source"].isin(exclude_sources)].copy()

        exclude_types = self.chunker_kwargs.get("exclude_event_types")
        if exclude_types:
            df = df[~df["event_type"].isin(exclude_types)].copy()

        df.reset_index(drop=True, inplace=True)

        stop_at = self.chunker_kwargs.get("stop_at")
        if stop_at:
            mask = pd.Series(True, index=df.index)
            for col, val in stop_at.items():
                mask &= df[col].astype(str).str.contains(val, regex=False)
            matches = df.index[mask]
            if len(matches) > 0:
                df = df.iloc[:matches[0] + 1].copy()
                df.reset_index(drop=True, inplace=True)

        return df

    def _event_index_to_step(self, eidx, boundaries: list[ChunkBoundary]) -> int | None:
        # LLM sometimes returns [N] instead of N
        if isinstance(eidx, list) and len(eidx) == 1:
            eidx = eidx[0]
        if eidx is None or not isinstance(eidx, (int, float)):
            return None
        eidx = int(eidx)
        for b in boundaries:
            if b.first_event_index <= eidx <= b.max_event_index:
                return b.step
        return 1 if eidx >= 0 else None

    def run(self, timeline_path: Path) -> ExtractionResult | None:
        """Extract features for one patient. Returns None on LLM failure."""
        hadm_id = timeline_path.stem.replace("timeline_", "")
        boundaries = self._get_boundaries(timeline_path)
        df = self._load_filtered(timeline_path)
        timeline_text = "\n".join(FORMATTER.format_events_numbered(df))

        prompt = USER_TEMPLATE.format(
            timeline=timeline_text,
            feature_list=self.feat_str,
            feature_json_example=self.feat_json,
        )

        raw = self._call_llm(SYSTEM_PROMPT, prompt)
        if raw is None:
            return None

        parsed = parse_llm_json(raw)
        if not parsed or not isinstance(parsed, dict):
            return None

        features = {}
        for fname in self.feat_names:
            entry = parsed.get(fname, {})
            if isinstance(entry, str):
                entry = {"value": entry, "event_index": None}

            val = entry.get("value", "not_stated")
            if val not in ("yes", "no", "not_stated"):
                val = "not_stated"

            eidx = entry.get("event_index")
            if val == "no" and eidx is None:
                val = "not_stated"

            step = self._event_index_to_step(eidx, boundaries)
            features[fname] = FeatureResult(value=val, event_index=eidx, step=step)

        return ExtractionResult(
            hadm_id=hadm_id,
            n_events=len(df),
            n_chunks=len(boundaries),
            boundaries=boundaries,
            features=features,
        )
