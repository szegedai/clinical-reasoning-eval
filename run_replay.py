"""CLI for running temporal diagnostic replay against an LLM.

Usage:
    python run_replay.py -c configs/replay_config.yaml --timeline-dir timelines/ -o results/run1
    python run_replay.py -c configs/replay_config.yaml --timeline-dir timelines/ -o results/run1 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_replay import PromptRenderer
from llm.runner import PatientRunner


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def build_runner(cfg: dict) -> PatientRunner:
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"Error: env var '{api_key_env}' is not set", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(
        base_url=cfg.get("base_url"),
        api_key=api_key,
    )

    prompts_cfg = cfg.get("prompts", {})
    renderer = PromptRenderer(
        system_prompt=prompts_cfg.get("system", "system_prompt.md"),
        step_prompt=prompts_cfg.get("step", "step_prompt.md"),
        step_prompt_cumulative=prompts_cfg.get("step_cumulative", "step_prompt_cumulative.md"),
        onepass_prompt=prompts_cfg.get("onepass", "onepass_prompt.md"),
        step_prompt_compressed=prompts_cfg.get("step_compressed", "step_prompt_compressed.md"),
        step_prompt_compressed_initial=prompts_cfg.get("step_compressed_initial", "step_prompt_compressed_initial.md"),
    )

    chunker_kwargs = {}
    chunker_cfg = cfg.get("chunker", {})
    if "max_events" in chunker_cfg:
        chunker_kwargs["max_events"] = chunker_cfg["max_events"]
    if "max_event_types" in chunker_cfg:
        chunker_kwargs["max_event_types"] = chunker_cfg["max_event_types"]
    if "max_hours" in chunker_cfg:
        chunker_kwargs["max_hours"] = chunker_cfg["max_hours"]
    if "stop_at" in chunker_cfg:
        chunker_kwargs["stop_at"] = chunker_cfg["stop_at"]
    if "exclude_sources" in chunker_cfg:
        chunker_kwargs["exclude_sources"] = set(chunker_cfg["exclude_sources"]) if chunker_cfg["exclude_sources"] else None
    if "exclude_event_types" in chunker_cfg:
        chunker_kwargs["exclude_event_types"] = set(chunker_cfg["exclude_event_types"]) if chunker_cfg["exclude_event_types"] else None
    if "max_chunks" in chunker_cfg:
        chunker_kwargs["max_chunks"] = chunker_cfg["max_chunks"]

    return PatientRunner(
        client=client,
        model=cfg["model"],
        renderer=renderer,
        chunker_kwargs=chunker_kwargs,
        temperature=cfg.get("temperature", 0.0),
        max_retries=cfg.get("max_retries", 3),
        max_steps=cfg.get("max_steps"),
        stop_after_confidence=cfg.get("stop_after_confidence"),
        mode=cfg.get("mode", "conversational"),
    )


def main():
    parser = argparse.ArgumentParser(description="Run temporal diagnostic replay against an LLM")
    parser.add_argument("--config", "-c", type=str, required=True, help="YAML config file")
    parser.add_argument("--timeline-dir", type=str, required=True, help="Directory of timeline CSVs")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for results")
    parser.add_argument("--skip-existing", action="store_true", help="Skip patients with existing result files")
    parser.add_argument("--limit", type=int, default=None, help="Max number of patients to process")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Reference results dir (required for onepass mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg.get("mode", "conversational")
    timeline_dir = Path(args.timeline_dir)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    if mode == "onepass" and reference_dir is None:
        print("Error: --reference-dir is required for onepass mode", file=sys.stderr)
        sys.exit(1)

    timeline_files = sorted(timeline_dir.glob("timeline_*.csv"))
    if args.limit:
        timeline_files = timeline_files[:args.limit]

    # For onepass, only process patients that have a reference result
    if mode == "onepass" and reference_dir is not None:
        filtered = []
        for tf in timeline_files:
            hadm_id = tf.stem.replace("timeline_", "")
            ref_file = reference_dir / f"patient_{hadm_id}.json"
            if ref_file.exists():
                filtered.append(tf)
        timeline_files = filtered

    if not timeline_files:
        print(f"No timeline_*.csv files found in {timeline_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = build_runner(cfg)

    # Write run_config at start
    run_config = {
        **cfg,
        "n_patients": len(timeline_files),
        "timeline_dir": str(timeline_dir),
        "reference_dir": str(reference_dir) if reference_dir else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(f"Batch run: {len(timeline_files)} patients → {out_dir}")
    print(f"Model: {cfg['model']} | Mode: {mode} | Temperature: {cfg.get('temperature', 0.0)}")
    print()

    failures = []
    n_skipped = 0

    for i, tf in enumerate(timeline_files, 1):
        hadm_id = tf.stem.replace("timeline_", "")
        out_file = out_dir / f"patient_{hadm_id}.json"

        if args.skip_existing and out_file.exists():
            n_skipped += 1
            print(f"[{i}/{len(timeline_files)}] {hadm_id} — skipped (exists)")
            continue

        try:
            if mode == "onepass":
                ref_file = reference_dir / f"patient_{hadm_id}.json"
                result = runner.run_onepass(tf, ref_file)
            else:
                result = runner.run(tf)
            out_file.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

            status = f"{len(result.steps)} steps, {result.total_input_tokens}+{result.total_output_tokens} tokens"
            print(f"[{i}/{len(timeline_files)}] {hadm_id} — {status}")

        except Exception as e:
            failures.append({"hadm_id": hadm_id, "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(timeline_files)}] {hadm_id} — FAILED: {failures[-1]['error']}", file=sys.stderr)

    # Update run_config with results
    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_config["completed"] = len(timeline_files) - len(failures) - n_skipped
    run_config["failures"] = failures
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(f"\nResults saved to: {out_dir}")
    if failures:
        print(f"{len(failures)} patient(s) failed — see run_config.json for details")


if __name__ == "__main__":
    main()
