"""
Queries BigQuery for hadm_ids matching a primary ICD code range (seq_num=1)

Usage:
  # Using pathology config (loads ICD-9+10 codes and HPI pattern from YAML):
  python create_cohort.py --pathology appendicitis --output appendicitis_ids.txt
  python create_cohort.py --pathology appendicitis --exclude-hpi-dx --output appendicitis_clean.txt

  # Require free-text discharge diagnosis to match pathology regexes:
  python create_cohort.py --pathology appendicitis --require-freetextdx --output appendicitis_ids.txt

  # Using explicit ICD prefixes (backwards compatible):
  python create_cohort.py --icd-range K35,K37 --output appendicitis_ids.txt
  python create_cohort.py --icd-range K35,K37 --exclude-hpi-dx --hpi-pattern "(?i)appendicitis" --output out.txt

  # Other options:
  python create_cohort.py --pathology appendicitis --no-require-note --no-require-ed --output cohort.txt
  python create_cohort.py --pathology appendicitis --dry-run
"""

import argparse
import sys
from pathlib import Path

import yaml
from google.cloud import bigquery

HOSP_DS = "physionet-data.mimiciv_3_1_hosp"
ED_DS = "physionet-data.mimiciv_ed"
NOTE_DS = "physionet-data.mimiciv_note"

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_pathology_config(name: str) -> dict:
    """Load a single pathology definition from pathologies.yaml."""
    yaml_path = CONFIGS_DIR / "pathologies.yaml"
    if not yaml_path.exists():
        sys.exit(f"Config file not found: {yaml_path}")
    with open(yaml_path) as f:
        all_pathologies = yaml.safe_load(f)
    if name not in all_pathologies:
        available = ", ".join(sorted(all_pathologies.keys()))
        sys.exit(f"Unknown pathology '{name}'. Available: {available}")
    return all_pathologies[name]


def build_cohort_query(
    icd_prefixes: list,
    require_note: bool,
    require_ed: bool,
    limit: int = None,
    exclude_hpi_pattern: str = None,
    require_freetextdx_patterns: list[str] = None,
) -> str:
    icd_conditions = " OR ".join(
        f"dx.icd_code LIKE '{prefix}%'" for prefix in icd_prefixes
    )

    needs_note = require_note or exclude_hpi_pattern or require_freetextdx_patterns
    note_join = ""
    if needs_note:
        note_join = f"""
  JOIN `{NOTE_DS}.discharge` n ON n.hadm_id = dx.hadm_id"""

    ed_join = ""
    if require_ed:
        ed_join = f"""
  JOIN `{ED_DS}.edstays` es ON es.hadm_id = dx.hadm_id"""

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    needs_cte = exclude_hpi_pattern or require_freetextdx_patterns
    if needs_cte:
        # CTE-based query: extract text sections, then filter
        cte_columns = ["dx.hadm_id"]
        where_clauses = []

        if exclude_hpi_pattern:
            cte_columns.append("""COALESCE(
      REGEXP_EXTRACT(
        REPLACE(n.text, '\\n', ' '),
        r'(?i)(?:history|___) of present(?:ing)? illness:(.+?)(?:physical exam(?:ination)?:|physical ___:|(?:pertinent|___) results:|hospital course:)'
      ),
      ''
    ) AS hpi""")
            where_clauses.append(
                f"NOT REGEXP_CONTAINS(hpi, r'{exclude_hpi_pattern}')"
            )

        if require_freetextdx_patterns:
            cte_columns.append("""COALESCE(
      REGEXP_EXTRACT(
        REPLACE(n.text, '\\n', ' '),
        r'(?i)(?:discharge|___) diagnosis:(.+?)(?:discharge condition|___ condition|condition:|procedure)'
      ),
      ''
    ) AS freetextdx""")
            combined = "|".join(
                f"(?:{p})" for p in require_freetextdx_patterns
            )
            where_clauses.append(
                f"REGEXP_CONTAINS(freetextdx, r'{combined}')"
            )

        cte_select = ",\n    ".join(cte_columns)
        filter_clause = "\n  AND ".join(where_clauses)

        return f"""
WITH base AS (
  SELECT DISTINCT
    {cte_select}
  FROM `{HOSP_DS}.diagnoses_icd` dx{note_join}{ed_join}
  WHERE dx.seq_num = 1
    AND ({icd_conditions})
)
SELECT hadm_id FROM base
WHERE {filter_clause}{limit_clause}
"""
    else:
        # Simple query without text filtering
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

    # Source of ICD codes: either --pathology or --icd-range
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--pathology", type=str,
        help="Pathology name from configs/pathologies.yaml (loads ICD-9+10 codes)",
    )
    source_group.add_argument(
        "--icd-range", type=str,
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
    parser.add_argument(
        "--exclude-hpi-dx", action="store_true",
        help="Exclude patients whose HPI mentions the disease name",
    )
    parser.add_argument(
        "--hpi-pattern", type=str, default=None,
        help="BQ re2 regex for disease name in HPI (required with --exclude-hpi-dx + --icd-range)",
    )
    parser.add_argument(
        "--require-freetextdx", action="store_true",
        help="Require discharge note free-text diagnosis to match pathology regexes (dx_match + dx_gracious)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the SQL and estimated bytes without executing",
    )
    args = parser.parse_args()

    # Resolve ICD prefixes and HPI pattern
    if args.pathology:
        cfg = load_pathology_config(args.pathology)
        prefixes = cfg.get("icd_10", []) + cfg.get("icd_9", [])
        hpi_pattern = cfg.get("hpi_exclude")
    else:
        prefixes = [p.strip() for p in args.icd_range.split(",")]
        hpi_pattern = args.hpi_pattern

    # Validate --exclude-hpi-dx usage
    exclude_pattern = None
    if args.exclude_hpi_dx:
        if not hpi_pattern:
            sys.exit("--exclude-hpi-dx requires either --pathology (auto) or --hpi-pattern (explicit)")
        exclude_pattern = hpi_pattern
        if not args.require_note:
            print("Note: --exclude-hpi-dx forces discharge note JOIN (overrides --no-require-note)")

    # Validate --require-freetextdx usage
    freetextdx_patterns = None
    if args.require_freetextdx:
        if not args.pathology:
            sys.exit("--require-freetextdx requires --pathology (loads dx_match + dx_gracious from YAML)")
        freetextdx_patterns = cfg.get("dx_match", []) + cfg.get("dx_gracious", [])
        if not freetextdx_patterns:
            sys.exit(f"No dx_match or dx_gracious patterns found for {args.pathology}")

    sql = build_cohort_query(
        icd_prefixes=prefixes,
        require_note=args.require_note,
        require_ed=args.require_ed,
        limit=args.limit,
        exclude_hpi_pattern=exclude_pattern,
        require_freetextdx_patterns=freetextdx_patterns,
    )

    print(f"ICD prefixes: {prefixes}")
    print(f"Require note: {args.require_note}")
    print(f"Require ED:   {args.require_ed}")
    if exclude_pattern:
        print(f"HPI exclude:  {exclude_pattern}")
    if freetextdx_patterns:
        print(f"Require freetextdx: {len(freetextdx_patterns)} patterns")

    client = bigquery.Client(project=args.project)

    if args.dry_run:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=job_config)
        gb = job.total_bytes_processed / 1e9
        print(f"\nEstimated scan: {gb:.2f} GB (${gb / 1000 * 5:.3f} at $5/TB)")
        print(f"\nSQL:\n{sql}")
        return

    print(f"\nSQL:\n{sql}")
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
