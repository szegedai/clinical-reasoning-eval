"""
Queries BigQuery for hadm_ids matching a primary ICD code range (seq_num=1)

Usage:
  python create_cohort.py --icd-range K35,K37 --output appendicitis_ids.txt
  python create_cohort.py --icd-range K35,K37 --no-require-note --no-require-ed --output cohort.txt
  python create_cohort.py --icd-range I21,I22 --limit 500 --output mi_ids.txt
"""

import argparse
import sys
from google.cloud import bigquery

HOSP_DS = "physionet-data.mimiciv_3_1_hosp"
ED_DS = "physionet-data.mimiciv_ed"
NOTE_DS = "physionet-data.mimiciv_note"


def build_cohort_query(
    icd_prefixes: list,
    require_note: bool,
    require_ed: bool,
    limit: int = None,
) -> str:
    icd_conditions = " OR ".join(
        f"dx.icd_code LIKE '{prefix}%'" for prefix in icd_prefixes
    )

    note_join = ""
    if require_note:
        note_join = f"""
  JOIN `{NOTE_DS}.discharge` dn ON dn.hadm_id = dx.hadm_id"""

    ed_join = ""
    if require_ed:
        ed_join = f"""
  JOIN `{ED_DS}.edstays` es ON es.hadm_id = dx.hadm_id"""

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""
SELECT DISTINCT dx.hadm_id
FROM `{HOSP_DS}.diagnoses_icd` dx{note_join}{ed_join}
WHERE dx.seq_num = 1
  AND ({icd_conditions}){limit_clause}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Create a cohort of hadm_ids by primary ICD code range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--icd-range", type=str, required=True,
        help="Comma-separated ICD code prefixes (e.g. K35,K37 for appendicitis)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output file path (one hadm_id per line)",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="GCP billing project ID",
    )
    parser.add_argument(
        "--no-require-note", dest="require_note", action="store_false", default=True,
        help="Don't require a discharge note (default: require)",
    )
    parser.add_argument(
        "--no-require-ed", dest="require_ed", action="store_false", default=True,
        help="Don't require ED admission (default: require)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of hadm_ids to return",
    )
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.icd_range.split(",")]
    sql = build_cohort_query(
        icd_prefixes=prefixes,
        require_note=args.require_note,
        require_ed=args.require_ed,
        limit=args.limit,
    )

    print(f"ICD prefixes: {prefixes}")
    print(f"Require note: {args.require_note}")
    print(f"Require ED:   {args.require_ed}")
    print(f"\nSQL:\n{sql}")

    client = bigquery.Client(project=args.project)
    print("Querying BigQuery...")
    df = client.query(sql).to_dataframe()

    if df.empty:
        sys.exit("No matching hadm_ids found.")

    hadm_ids = sorted(df["hadm_id"].tolist())
    with open(args.output, "w") as f:
        for h in hadm_ids:
            f.write(f"{h}\n")

    print(f"\nFound {len(hadm_ids)} hadm_ids. Saved to: {args.output}")


if __name__ == "__main__":
    main()
