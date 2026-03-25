"""Collect per-step results from a replay run into an Excel file.

Usage:
    python collect_results.py results/gemini-2.0-flash_2026-03-15T19-01-12/
    python collect_results.py results/gemini-2.0-flash_2026-03-15T19-01-12/ -o my_results.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Collect replay results into Excel")
    parser.add_argument("run_dir", help="Directory with patient_*.json files")
    parser.add_argument("-o", "--output", help="Output Excel path (default: <run_dir>/results.xlsx)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    files = sorted(run_dir.glob("patient_*.json"))
    if not files:
        print(f"No patient_*.json files in {run_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        data = json.loads(f.read_text())
        for step in data["steps"]:
            parsed = step.get("parsed") or {}
            # Top diagnosis from differential
            diff = parsed.get("differential") or []
            top_dx = diff[0]["diagnosis"] if diff else None
            top_conf = diff[0]["confidence"] if diff else None

            rows.append({
                "hadm_id": data["hadm_id"],
                "step": step["step"],
                "label": step["label"],
                "n_events": step["n_events"],
                "assessment": parsed.get("assessment"),
                "top_diagnosis": top_dx,
                "top_confidence": top_conf,
                "n_differentials": len(diff),
                "key_findings": str(parsed.get("key_findings") or ""),
                "input_tokens": step["input_tokens"],
                "output_tokens": step["output_tokens"],
                "latency_ms": step["latency_ms"],
                "parse_error": parsed.get("parse_error"),
            })

    df = pd.DataFrame(rows)
    out_path = args.output or str(run_dir / "results.xlsx")
    df.to_excel(out_path, index=False)
    print(f"{len(df)} rows from {len(files)} patients → {out_path}")


if __name__ == "__main__":
    main()
