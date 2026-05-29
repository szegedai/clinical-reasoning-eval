"""Cohort loader: find and filter patient timeline paths."""
from __future__ import annotations

from pathlib import Path


def load_timelines(pathology_dir: Path) -> list[Path]:
    paths = sorted(pathology_dir.glob("timeline_*.csv"))
    return paths


def filter_by_ids(
    paths: list[Path],
    hadm_ids: list[int | str],
) -> list[Path]:
    id_set = {str(i) for i in hadm_ids}
    return [p for p in paths if p.stem.replace("timeline_", "") in id_set]


def load_from_id_file(
    pathology_dir: Path,
    id_file: Path,
) -> list[Path]:
    hadm_ids = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
    all_paths = load_timelines(pathology_dir)
    return filter_by_ids(all_paths, hadm_ids)
