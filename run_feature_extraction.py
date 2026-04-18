"""CLI for extracting clinical features with temporal event anchoring.

Usage:
    # Extract (one JSON per patient, resumable)
    python run_feature_extraction.py extract \
        -c configs/replay_config_gemini.yaml \
        --features configs/feature_list.json \
        --timeline-dir /path/to/timelines \
        -o /path/to/output

    # Build temporal feature matrix from extracts
    python run_feature_extraction.py build \
        --features configs/feature_list.json \
        -o /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openai
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_extraction import FeatureExtractor, build_temporal_matrix
from temporal_replay import TooManyChunksError, NegativeElapsedTimeError


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def parse_chunker_kwargs(cfg: dict) -> dict:
    ck = cfg.get("chunker", {})
    kwargs = {}
    for key in ("max_events", "max_event_types", "max_hours", "stop_at", "max_chunks"):
        if key in ck:
            kwargs[key] = ck[key]
    if "exclude_sources" in ck:
        kwargs["exclude_sources"] = set(ck["exclude_sources"]) if ck["exclude_sources"] else None
    if "exclude_event_types" in ck:
        kwargs["exclude_event_types"] = set(ck["exclude_event_types"]) if ck["exclude_event_types"] else None
    return kwargs


def cmd_extract(args):
    cfg = load_config(args.config)
    features = load_features(args.features)

    api_key_env = cfg.get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        sys.exit(f"Set {api_key_env} environment variable")

    client = openai.OpenAI(base_url=cfg.get("base_url"), api_key=api_key)

    extractor = FeatureExtractor(
        client=client,
        model=cfg["model"],
        features=features,
        chunker_kwargs=parse_chunker_kwargs(cfg),
        temperature=cfg.get("temperature", 0.0),
        max_retries=cfg.get("max_retries", 3),
    )

    timeline_dir = Path(args.timeline_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_files = sorted(timeline_dir.glob("timeline_*.csv"))
    if args.limit:
        timeline_files = timeline_files[:args.limit]

    existing = {p.stem.replace("extract_", "") for p in out_dir.glob("extract_*.json")}
    todo = [(tf, tf.stem.replace("timeline_", ""))
            for tf in timeline_files
            if tf.stem.replace("timeline_", "") not in existing]

    print(f"Patients: {len(timeline_files)}, already done: {len(existing)}, "
          f"remaining: {len(todo)}")
    print(f"Features: {len(features)}, model: {cfg['model']}")

    extracted, errors, skipped = 0, 0, 0

    for i, (tf, hadm_id) in enumerate(todo):
        try:
            result = extractor.run(tf)
        except (TooManyChunksError, NegativeElapsedTimeError):
            skipped += 1
            continue

        if result is None:
            errors += 1
            print(f"  [{i+1}/{len(todo)}] {hadm_id} — failed")
            continue

        out_file = out_dir / f"extract_{hadm_id}.json"
        out_file.write_text(json.dumps(result.to_dict(), indent=2))
        extracted += 1

        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] extracted={extracted} errors={errors} "
                  f"skipped={skipped}")

    print(f"\nDone. extracted={extracted}, errors={errors}, skipped={skipped}")


def cmd_build(args):
    features = load_features(args.features)
    feat_names = [f["name"] for f in features]

    out_dir = Path(args.output)
    matrix_path = out_dir / "temporal_feature_matrix.csv"

    df = build_temporal_matrix(out_dir, feat_names, matrix_path)
    print(f"Built: {len(df)} rows ({df['hadm_id'].nunique()} patients)")
    print(f"Saved: {matrix_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract clinical features with temporal event anchoring")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("-c", "--config", required=True)
    p_extract.add_argument("--features", required=True)
    p_extract.add_argument("--timeline-dir", required=True)
    p_extract.add_argument("-o", "--output", required=True)
    p_extract.add_argument("--limit", type=int, default=None)

    p_build = sub.add_parser("build")
    p_build.add_argument("--features", required=True)
    p_build.add_argument("-o", "--output", required=True)

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "build":
        cmd_build(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
