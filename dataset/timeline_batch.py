#!/usr/8in/env python3
"""
Usage:
  python timeline_batch.py --file appendicitis_ids.txt --output-dir timelines
  python timeline_batch.py --file appendicitis_ids.txt --output-dir timelines --batch-size 200
"""

import argparse
import traceback

import sys
import os
from typing import List

from google.cloud import bigquery
import pandas as pd
from timeline import build_query, parse_discharge_note_sections, sort_timeline


def get_timeline_batch(
    hadm_ids: List[int],
    project: str = None,
) -> pd.DataFrame:
    client = bigquery.Client(project=project)

    sql = build_query()

    sql = sql.replace("= @hadm_id", "IN UNNEST(@hadm_ids)")

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
        ]
    )

    print(f"Querying BQ for batch of {len(hadm_ids)} admissions...")
    job = client.query(sql, job_config=job_config)
    df = job.to_dataframe()

    if df.empty:
        print("WARNING: No rows returned for batch.")
        return df

    bytes_billed = job.total_bytes_billed or 0
    print(f"  Rows returned : {len(df):,}")
    print(f"  Bytes billed  : {bytes_billed / 1e6:.1f} MB")

    notes_sql = f"""
    SELECT hadm_id, charttime, text
    FROM `physionet-data.mimiciv_note.discharge`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    notes_job = client.query(notes_sql, job_config=job_config)
    notes_raw = notes_job.to_dataframe()

    _valid = df[df["time_precision"] != "date_only"]
    t0_map = _valid.groupby("hadm_id")["event_time"].min()
    subj_map = df.groupby("hadm_id")["subject_id"].first()

    all_note_rows = []
    for _, row in notes_raw.iterrows():
        if not row["text"]:
            continue
        try:
            h_id = row["hadm_id"]
            t0 = t0_map.get(h_id, row["charttime"])
            all_note_rows.extend(parse_discharge_note_sections(
                hadm_id=h_id,
                text=row["text"],
                note_time=row["charttime"],
                t0=t0,
                subject_id=subj_map.get(h_id),
            ))
        except Exception as e:
            print(f"Error parsing notes for hadm_id {row['hadm_id']}: {e}")
            traceback.print_exc()

    if all_note_rows:
        note_df = pd.DataFrame(all_note_rows)
        df = pd.concat([df, note_df], ignore_index=True)
        print(f"  Added {len(note_df)} discharge note section rows.")

    return df

def main():
    parser = argparse.ArgumentParser(description="Batch process MIMIC-IV timelines.")
    parser.add_argument("--file", type=str, required=True, help="File containing one hadm_id per line")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save CSVs")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=200, help="Number of hadm_ids per BQ query")
    args = parser.parse_args()

    ids = []
    with open(args.file, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                ids.append(int(line))

    if not ids:
        sys.exit(f"Error: No hadm_ids found in {args.file}.")

    ids = list(set(ids))
    os.makedirs(args.output_dir, exist_ok=True)

    pending_ids = []
    for h in ids:
        if not os.path.exists(os.path.join(args.output_dir, f"timeline_{h}.csv")):
            pending_ids.append(h)

    print(f"Total target hadm_ids: {len(ids)}")
    print(f"Already completed    : {len(ids) - len(pending_ids)}")
    print(f"Remaining to process : {len(pending_ids)}")

    if not pending_ids:
        print("All done!")
        return

    for i in range(0, len(pending_ids), args.batch_size):
        batch = pending_ids[i:i + args.batch_size]
        print(f"\n--- Processing batch {i//args.batch_size + 1} ({len(batch)} items) ---")

        df_batch = get_timeline_batch(
            hadm_ids=batch,
            project=args.project,
        )

        if df_batch.empty:
            continue

        for h, group in df_batch.groupby("hadm_id"):
            out_file = os.path.join(args.output_dir, f"timeline_{h}.csv")
            group = sort_timeline(group)
            group.to_csv(out_file, index=False)

        print(f"Saved {len(df_batch['hadm_id'].unique())} timeline CSVs to {args.output_dir}/")

if __name__ == "__main__":
    main()
