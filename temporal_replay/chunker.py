"""
Splits a chronological timeline CSV into clinical chunks,
enabling incremental presentation to an LLM for diagnostic reasoning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd


class NegativeElapsedTimeError(ValueError):
    """Raised when a timeline contains events with negative elapsed_hours."""

    def __init__(self, hadm_id: int, n_negative: int, min_hours: float):
        self.hadm_id = hadm_id
        self.n_negative = n_negative
        self.min_hours = min_hours
        super().__init__(
            f"hadm_id={hadm_id}: {n_negative} event(s) with negative elapsed_hours "
            f"(min={min_hours:.2f}h). Pre-admission events not supported."
        )


class TooManyChunksError(ValueError):
    """Raised when a timeline would produce more chunks than the allowed limit."""

    def __init__(self, hadm_id: int, n_chunks: int, max_chunks: int):
        self.hadm_id = hadm_id
        self.n_chunks = n_chunks
        self.max_chunks = max_chunks
        super().__init__(
            f"hadm_id={hadm_id}: timeline would produce {n_chunks} chunks "
            f"(max_chunks={max_chunks}). Skipping to avoid excessive cost."
        )


_STEP1_TYPES = {"ED_ARRIVAL", "DISCHARGE_HPI"}
_STEP2_TYPES = {
    "TRIAGE_COMPLAINT", "TRIAGE_PAIN", "TRIAGE_VITAL",
    "TRIAGE_ACUITY", "ED_VITALS",
}
_STEP3_TYPES = {"DISCHARGE_PE"}


@dataclass
class ReplayChunk:
    """A single chunk returned during replay iteration."""
    step: int
    events: pd.DataFrame          # events in this chunk
    history: pd.DataFrame         # all events up to (but not including) this chunk
    cumulative: pd.DataFrame      # all events up to and including this chunk
    label: str = ""               # human-readable label for this step


class TimelineChunker:
    """
    Iterate over a patient timeline in clinical chunks.

    Parameters
    ----------
    folder : str or Path
        Directory containing timeline CSVs.
    filename : str
        CSV filename (e.g. "timeline_20022233.csv").
    max_events : int
        Approximate max number of events per chunk (soft cap). Default 25.
    max_event_types : int
        Max distinct event types per chunk (soft cap). Default 3.
    max_hours : float
        Dynamic chunk fires when elapsed time within window exceeds this. Default 4.0.
    exclude_sources : set of str or None
        Source prefixes to exclude (e.g. {"ICU"} removes ICU_VITAL, ICU_INPUT, etc.).
    exclude_event_types : set of str or None
        Event types to exclude (e.g. {"DISCHARGE_DX"} removes diagnosis-revealing events).
    """

    def __init__(
        self,
        folder: str | Path,
        filename: str,
        *,
        max_events: int = 25,
        max_event_types: int = 3,
        max_hours: float = 4.0,
        stop_at: dict[str, str] | None = None,
        exclude_sources: set[str] | None = {"ICU"},
        exclude_event_types: set[str] | None = {"DISCHARGE_DX", "DISCHARGE_FREETEXTDX"},
        max_chunks: int | None = None,
    ):
        self.path = Path(folder) / filename
        if not self.path.exists():
            raise FileNotFoundError(f"Timeline not found: {self.path}")

        self.max_events = max_events
        self.max_event_types = max_event_types
        self.max_hours = max_hours
        self.stop_at = stop_at
        self.exclude_sources = exclude_sources
        self.exclude_event_types = exclude_event_types
        self.max_chunks = max_chunks

        self._df = self._load()

        # Pre-flight check: reject timelines that would produce too many chunks
        if self.max_chunks is not None:
            n_chunks = len(self._build_chunks())
            if n_chunks > self.max_chunks:
                raise TooManyChunksError(
                    hadm_id=self.hadm_id,
                    n_chunks=n_chunks,
                    max_chunks=self.max_chunks,
                )

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["event_time"])
        # Drop date_only events, they are not part of the timeline
        df = df[df["time_precision"] != "date_only"].copy()
        df.reset_index(drop=True, inplace=True)

        # Filter out excluded sources (e.g. ICU events)
        if self.exclude_sources:
            df = df[~df["source"].isin(self.exclude_sources)].copy()
            df.reset_index(drop=True, inplace=True)

        # Filter out excluded event types (e.g. discharge diagnoses)
        if self.exclude_event_types:
            df = df[~df["event_type"].isin(self.exclude_event_types)].copy()
            df.reset_index(drop=True, inplace=True)

        neg = df["elapsed_hours"] < 0
        if neg.any():
            raise NegativeElapsedTimeError(
                hadm_id=int(df["hadm_id"].iloc[0]),
                n_negative=int(neg.sum()),
                min_hours=float(df.loc[neg, "elapsed_hours"].min()),
            )

        # Truncate at stop_at event (inclusive)
        # Diagnosis was likely already captured at this point
        if self.stop_at:
            mask = pd.Series(True, index=df.index)
            for col, val in self.stop_at.items():
                mask &= df[col].astype(str).str.contains(val, regex=False)
            matches = df.index[mask]
            if len(matches) > 0:
                df = df.iloc[: matches[0] + 1].copy()
                df.reset_index(drop=True, inplace=True)

        return df

    @property
    def hadm_id(self) -> int:
        return int(self._df["hadm_id"].iloc[0])

    @property
    def subject_id(self) -> int:
        return int(self._df["subject_id"].iloc[0])

    @property
    def total_events(self) -> int:
        return len(self._df)

    def _build_chunks(self) -> list[tuple[str, pd.DataFrame]]:
        """Split the timeline into labeled chunks (label, DataFrame)."""
        df = self._df
        chunks: list[tuple[str, pd.DataFrame]] = []

        # Fixed initial steps (steps 1-3):
        #   - step 1: HPI & Arrival
        #   - step 2: Triage
        #   - step 3: Physical Exam
        # All three cover t=0 events only. Events that don't match any
        # fixed type are held back for dynamic chunking.

        s1_rows, s2_rows, s3_rows = [], [], []
        other_rows = []

        # Find the end of the initial window: last row with elapsed_hours == 0
        initial_end = 0
        for i in range(len(df)):
            if df.iloc[i]["elapsed_hours"] == 0:
                initial_end = i + 1
            else:
                break

        for i in range(initial_end):
            et = df.iloc[i]["event_type"]
            if et in _STEP1_TYPES:
                s1_rows.append(i)
            elif et in _STEP2_TYPES:
                s2_rows.append(i)
            elif et in _STEP3_TYPES:
                s3_rows.append(i)
            else:
                other_rows.append(i)

        if s1_rows:
            chunks.append(("HPI & Arrival", df.iloc[s1_rows]))
        if s2_rows:
            chunks.append(("Triage", df.iloc[s2_rows]))
        if s3_rows:
            chunks.append(("Physical Exam", df.iloc[s3_rows]))

        # Remaining = non-fixed events from the initial window + everything after
        remaining = pd.concat(
            [df.iloc[other_rows], df.iloc[initial_end:]],
            ignore_index=False,
        ) if other_rows else df.iloc[initial_end:]

        # --- Dynamic chunking ---
        if not remaining.empty:
            remaining = remaining.reset_index(drop=True)
            dynamic_chunks = self._dynamic_chunk(remaining)
            for i, dc in enumerate(dynamic_chunks, 1):
                t_start = dc["elapsed_hours"].min()
                t_end = dc["elapsed_hours"].max()
                if pd.notna(t_start) and pd.notna(t_end):
                    label = f"Events {t_start:.1f}h – {t_end:.1f}h"
                else:
                    label = f"Dynamic chunk {i}"
                chunks.append((label, dc))

        return chunks

    def _dynamic_chunk(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Split remaining events into chunks"""
        chunks: list[pd.DataFrame] = []
        chunk_start = 0
        n = len(df)

        # Each chunk grows until any threshold is exceeded:
        #   - max_events:      too many events in the window
        #   - max_event_types: too many distinct event types
        #   - max_hours:       too much time elapsed since the chunk started
        # Cut is "soft": once triggered, the boundary extends to include all
        # remaining events sharing the same timestamp (_finish_timestamp_group).

        while chunk_start < n:
            cut = chunk_start
            window_types: set[str] = set()
            t_start = df.iloc[chunk_start]["elapsed_hours"]

            for i in range(chunk_start, n):
                row = df.iloc[i]
                events_in_window = i - chunk_start + 1
                window_types.add(row["event_type"])
                t_elapsed = (row["elapsed_hours"] - t_start) if pd.notna(row["elapsed_hours"]) and pd.notna(t_start) else 0

                should_cut = (
                    events_in_window > self.max_events
                    or len(window_types) > self.max_event_types
                    or t_elapsed > self.max_hours
                )

                if should_cut and events_in_window > 1:
                    # Soft boundary: finish the current timestamp group
                    cut = self._finish_timestamp_group(df, i, n)
                    break
                cut = i + 1

            if cut <= chunk_start:
                cut = n

            chunk_df = df.iloc[chunk_start:cut].copy()
            chunks.append(chunk_df)
            chunk_start = cut

        return chunks

    @staticmethod
    def _finish_timestamp_group(df: pd.DataFrame, trigger_idx: int, n: int) -> int:
        """Extend past trigger_idx to include all rows with the same event_time."""
        trigger_time = df.iloc[trigger_idx]["event_time"]
        end = trigger_idx + 1
        while end < n and df.iloc[end]["event_time"] == trigger_time:
            end += 1
        return end

    def replay(self) -> Iterator[ReplayChunk]:
        """Iterate over timeline chunks. Each chunk exposes:
          - chunk.events — new events in this step
          - chunk.history — all events before this step
          - chunk.cumulative — all events up to and including this step
        """
        raw_chunks = self._build_chunks()
        history_frames: list[pd.DataFrame] = []

        for step_num, (label, chunk_df) in enumerate(raw_chunks, 1):
            history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
            cumulative = pd.concat(history_frames + [chunk_df], ignore_index=True)

            yield ReplayChunk(
                step=step_num,
                events=chunk_df.reset_index(drop=True),
                history=history,
                cumulative=cumulative,
                label=label,
            )

            history_frames.append(chunk_df)

    def chunk_summary(self) -> pd.DataFrame:
        """Return a summary table of all chunks"""
        raw_chunks = self._build_chunks()
        rows = []
        cum_count = 0
        for i, (label, chunk_df) in enumerate(raw_chunks, 1):
            cum_count += len(chunk_df)
            t_min = chunk_df["elapsed_hours"].min()
            t_max = chunk_df["elapsed_hours"].max()
            types = sorted(chunk_df["event_type"].unique())
            rows.append({
                "step": i,
                "label": label,
                "n_events": len(chunk_df),
                "cumulative_events": cum_count,
                "t_start_h": round(t_min, 2) if pd.notna(t_min) else None,
                "t_end_h": round(t_max, 2) if pd.notna(t_max) else None,
                "event_types": ", ".join(types),
            })
        return pd.DataFrame(rows)
