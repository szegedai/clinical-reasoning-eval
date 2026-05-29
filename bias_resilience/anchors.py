"""Anchor resolution: map anchor names to step indices for one patient.

  post_pe         — first step with chunker label "Physical Exam"
  post_first_labs — first step with a LAB_RESULT event (not SPECIMEN_COLLECTED)
  post_imaging    — first step with a RADIOLOGY_REPORT event (not IMAGING_STUDY)
"""
from __future__ import annotations

from pathlib import Path

from temporal_replay.chunker import TimelineChunker


def resolve_anchors(
    timeline_path: Path,
    chunker_kwargs: dict | None = None,
) -> dict[str, int | None]:
    ckw = dict(chunker_kwargs) if chunker_kwargs else {}
    chunker = TimelineChunker(str(timeline_path.parent), timeline_path.name, **ckw)

    post_pe: int | None = None
    post_first_labs: int | None = None
    post_imaging: int | None = None

    for chunk in chunker.replay():
        if post_pe is None and chunk.label == "Physical Exam":
            post_pe = chunk.step

        if post_first_labs is None:
            if "LAB_RESULT" in chunk.events["event_type"].values:
                post_first_labs = chunk.step

        if post_imaging is None:
            if "RADIOLOGY_REPORT" in chunk.events["event_type"].values:
                post_imaging = chunk.step

        if post_pe is not None and post_first_labs is not None and post_imaging is not None:
            break

    return {
        "post_pe": post_pe,
        "post_first_labs": post_first_labs,
        "post_imaging": post_imaging,
    }


def resolve_anchors_batch(
    timeline_paths: list[Path],
    chunker_kwargs: dict | None = None,
) -> dict[str, dict[str, int | None]]:
    results = {}
    for path in timeline_paths:
        patient_id = path.stem.replace("timeline_", "")
        results[patient_id] = resolve_anchors(path, chunker_kwargs)
    return results
