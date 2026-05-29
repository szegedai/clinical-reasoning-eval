"""CLI entrypoint for bias-resilience experiment runs.

Usage examples:
  # Smoke test: 1 patient, baseline, max 2 steps
  python -m bias_resilience.cli \\
    --run-id 2026-04-30_smoke \\
    --model gemini-2.5-flash \\
    --condition baseline \\
    --pathology cholecystitis \\
    --patients 1 \\
    --max-steps 2

  # Pinned cohort (run-id matches DEFAULT_COHORTS — manifest used automatically)
  python -m bias_resilience.cli \\
    --run-id m9_cholecystitis_post_pe \\
    --model gemini-2.5-flash \\
    --condition baseline \\
    --pathology cholecystitis

  # Explicit cohort file
  python -m bias_resilience.cli \\
    --run-id m9_cholecystitis_post_imaging \\
    --model gemini-2.5-flash \\
    --condition struc_belief \\
    --pathology cholecystitis \\
    --cohort-file bias_resilience/cohorts/m9_cholecystitis.txt \\
    --anchor post_imaging \\
    --max-hours 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    MODELS, CONDITIONS, TIMELINES_ROOT, RESULTS_ROOT,
    PATHOLOGY_CORRECT_LABEL, DEFAULT_COHORTS, REPO_ROOT, DEFAULT_CHUNKER_KWARGS,
)
from .runner import run_patient
from .cohort import load_timelines, load_from_id_file
from .anchors import resolve_anchors

_EXCLUDED_STATUSES = {"anchor_not_resolved", "anchor_past_cap"}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m bias_resilience.cli",
        description="Run bias-resilience experiments on MIMIC timelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", required=True,
                   help="Human-readable run identifier (e.g. '2026-04-30_phase1_n10').")
    p.add_argument("--model", required=True, choices=list(MODELS),
                   help="Model key from config.MODELS.")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS),
                   help="Condition name from config.CONDITIONS.")
    p.add_argument("--pathology", required=True,
                   help="Pathology subdirectory under TIMELINES_ROOT (e.g. 'cholecystitis').")
    p.add_argument("--patients", type=int, default=None,
                   help="Max number of patients (default: all). Ignored when --cohort-file is set.")
    p.add_argument("--cohort-file", default=None,
                   help="Path to a pinned cohort manifest (one patient_id per line). "
                        "Replaces filesystem-order slicing. Warn if --patients is also set.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard cap on replay steps per patient (use ≤2 for smoke tests).")
    p.add_argument("--max-hours", type=float, default=48.0,
                   help="Duration cap: skip chunks past this many hours from admission "
                        "(default 48.0; pass 0 to disable).")
    p.add_argument("--workers", type=int, default=16,
                   help="ThreadPoolExecutor workers (default 16; raise to 128 for large remote batches).")
    p.add_argument("--anchor", default=None,
                   help="Override anchor (post_pe | post_first_labs | post_imaging). "
                        "Defaults to all anchors configured for the condition.")
    p.add_argument("--timelines-root", default=None,
                   help="Override TIMELINES_ROOT from config.")
    p.add_argument("--results-root", default=None,
                   help="Override RESULTS_ROOT from config.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output already exists.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    model_cfg = MODELS[args.model]
    condition = args.condition
    timelines_root = Path(args.timelines_root) if args.timelines_root else TIMELINES_ROOT
    results_root = Path(args.results_root) if args.results_root else RESULTS_ROOT

    pathology_dir = timelines_root / args.pathology
    if not pathology_dir.exists():
        print(f"ERROR: pathology directory not found: {pathology_dir}", file=sys.stderr)
        sys.exit(1)

    # Cohort selection — explicit manifest > default manifest > filesystem slice
    if args.cohort_file is not None and args.patients is not None:
        print("WARNING: --cohort-file and --patients both specified; --patients is ignored.",
              file=sys.stderr)

    if args.cohort_file is not None:
        cohort_path = Path(args.cohort_file)
        if not cohort_path.exists():
            print(f"ERROR: cohort manifest not found: {cohort_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Cohort source: explicit manifest {cohort_path}")
        all_paths = load_from_id_file(pathology_dir, cohort_path)
    elif args.run_id in DEFAULT_COHORTS:
        cohort_path = REPO_ROOT / DEFAULT_COHORTS[args.run_id]
        if not cohort_path.exists():
            print(f"ERROR: default cohort manifest not found: {cohort_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Cohort source: default manifest {cohort_path} (run-id match)")
        all_paths = load_from_id_file(pathology_dir, cohort_path)
    else:
        all_paths = load_timelines(pathology_dir)
        if args.patients is not None:
            all_paths = all_paths[: args.patients]
        print(f"Cohort source: filesystem order, first {len(all_paths)}")

    if not all_paths:
        print(f"ERROR: no timeline CSVs found for the selected cohort in {pathology_dir}",
              file=sys.stderr)
        sys.exit(1)

    max_hours: float | None = None if args.max_hours <= 0 else args.max_hours

    print(f"Patients: {len(all_paths)}")
    print(f"Model:     {model_cfg.name} ({model_cfg.model_id})")
    print(f"Condition: {condition}")
    print(f"Max steps: {args.max_steps or 'all'}")
    print(f"Max hours: {max_hours if max_hours is not None else 'disabled'}")

    chunker_kwargs = DEFAULT_CHUNKER_KWARGS

    print()

    # Determine anchors for this condition
    cond_cfg = CONDITIONS[condition]
    anchors: list[str | None] = cond_cfg["anchors"] if cond_cfg["anchors"] else [None]
    if args.anchor is not None:
        anchors = [args.anchor]

    completed = 0
    excluded = 0
    failed = 0
    total_cost = 0.0

    for anchor in anchors:
        anchor_label = anchor or "no_anchor"
        print(f"--- Anchor: {anchor_label} ---")
        for tp in all_paths:
            patient_id = tp.stem.replace("timeline_", "")
            try:
                anchor_step = None
                if anchor is not None and condition != "baseline":
                    try:
                        patient_anchors = resolve_anchors(tp)
                        anchor_step = patient_anchors.get(anchor)
                    except Exception as e:
                        print(f"  {patient_id}: anchor resolution error — {e}",
                              file=sys.stderr)
                        failed += 1
                        continue

                correct_dx = PATHOLOGY_CORRECT_LABEL.get(args.pathology)
                if correct_dx is None and args.pathology:
                    print(f"WARNING: no correct_dx label for pathology '{args.pathology}'; "
                          "struc_belief WD will use generic placeholder.", file=sys.stderr)
                summary = run_patient(
                    timeline_path=tp,
                    patient_id=patient_id,
                    run_id=args.run_id,
                    model_cfg=model_cfg,
                    condition=condition,
                    anchor=anchor,
                    anchor_step=anchor_step,
                    pathology=args.pathology,
                    correct_dx=correct_dx,
                    results_root=results_root,
                    force=args.force,
                    max_steps=args.max_steps,
                    max_hours=max_hours,
                    chunker_kwargs=chunker_kwargs,
                )
                status = summary.get("status", "?")
                cost = summary.get("cost_usd_total", 0.0)
                total_cost += cost
                print(f"  {patient_id}: {status}  (${cost:.4f})")
                if status == "ok":
                    completed += 1
                elif status in _EXCLUDED_STATUSES:
                    excluded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  {patient_id}: EXCEPTION — {e}", file=sys.stderr)
                failed += 1

    print(f"\nDone. completed={completed}  excluded={excluded}  failed={failed}  "
          f"actual_cost=${total_cost:.4f}")


if __name__ == "__main__":
    main()
