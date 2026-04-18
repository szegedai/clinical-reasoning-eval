"""Build temporal feature matrix from per-patient extraction JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_temporal_matrix(
    extracts_dir: Path,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Read extract JSONs and produce a temporal feature matrix.

    One row per (patient, step). Each feature is "yes"/"no" if its anchoring
    event has appeared by that step, otherwise "not_stated".
    """
    extract_files = sorted(extracts_dir.glob("extract_*.json"))
    if not extract_files:
        raise FileNotFoundError(f"No extract_*.json files in {extracts_dir}")

    rows = []
    for fp in extract_files:
        data = json.loads(fp.read_text())
        for b in data["boundaries"]:
            step = b["step"]
            row = {
                "hadm_id": data["hadm_id"],
                "step": step,
                "step_label": b["label"],
                "n_cumulative_events": b["max_event_index"] + 1,
            }
            for fname in feature_names:
                info = data["features"].get(fname, {})
                feat_step = info.get("step")
                val = info.get("value", "not_stated")
                row[fname] = val if (feat_step is not None and feat_step <= step) else "not_stated"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
