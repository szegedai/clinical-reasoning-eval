"""Create a diverse 'other abdominal' cohort from 25 diagnosis groups.

For each group: query BQ for matching hadm_ids, sample N per group.
After timeline extraction, a separate step filters out HPI leaks.

Usage:
    python bayes/create_other_cohort.py --dry-run
    python bayes/create_other_cohort.py --per-group 30
"""

import argparse
import sys

import numpy as np
from google.cloud import bigquery

HOSP_DS = "physionet-data.mimiciv_3_1_hosp"
ED_DS = "physionet-data.mimiciv_ed"
NOTE_DS = "physionet-data.mimiciv_note"

# ICD codes for the 4 target pathologies (excluded from "other")
EXCLUDE_TARGET = """
    (dx.icd_version = 10 AND (dx.icd_code LIKE 'K35%' OR dx.icd_code LIKE 'K37%'))
    OR (dx.icd_version = 9 AND (dx.icd_code LIKE '540%' OR dx.icd_code LIKE '541%' OR dx.icd_code LIKE '542%'))
    OR (dx.icd_version = 10 AND dx.icd_code LIKE 'K85%')
    OR (dx.icd_version = 9 AND dx.icd_code LIKE '5770%')
    OR (dx.icd_version = 10 AND dx.icd_code LIKE 'K81%')
    OR (dx.icd_version = 9 AND (dx.icd_code LIKE '5750%' OR dx.icd_code LIKE '57510%' OR dx.icd_code LIKE '57511%' OR dx.icd_code LIKE '57512%'))
    OR (dx.icd_version = 10 AND dx.icd_code IN ('K5700','K5712','K5713','K5720','K5721','K5732','K5733','K5740','K5752','K5780','K5781','K5792','K5793'))
    OR (dx.icd_version = 9 AND dx.icd_code IN ('56201','56203','56211','56213'))
"""

# 25 diagnosis groups: (name, icd_prefixes_v10, icd_prefixes_v9, hpi_exclude_pattern)
GROUPS = [
    ("intestinal_obstruction",
     ["K566", "K565", "K560"],
     ["5609", "56081", "56089"],
     r"obstruction|ileus|\bSBO\b|bowel obstruct"),

    ("sepsis",
     ["A41", "A40"],
     ["0389"],
     r"sepsis|septic"),

    ("gastroenteritis",
     ["A084", "A09", "K529"],
     ["5589", "0088"],
     r"gastroenteritis|viral enteritis"),

    ("c_diff",
     ["A047"],
     ["00845"],
     r"difficile|C\.?\s*diff|\bCDIFF\b"),

    ("cholangitis",
     ["K830"],
     ["5761"],
     r"cholangitis"),

    ("choledocholithiasis",
     ["K805"],
     ["57451"],
     r"choledocholithiasis|CBD stone|bile duct stone|common bile duct stone"),

    ("gallstones_no_cholecystitis",
     ["K8010", "K8012", "K800"],
     ["57400", "57410"],
     r"cholelithiasis|gallstone|gallbladder stone|biliary colic"),

    ("constipation",
     ["K5900"],
     ["56400"],
     r"constipation"),

    ("crohns",
     ["K50"],
     ["555"],
     r"crohn|regional enteritis"),

    ("ulcerative_colitis",
     ["K51"],
     ["556"],
     r"ulcerative colitis|\bUC\b"),

    ("gi_bleed",
     ["K922"],
     ["5789"],
     r"GI bleed|gastrointestinal bleed|melena|hematochezia|\bGIB\b"),

    ("alcoholic_liver",
     ["K70"],
     ["5712", "5713"],
     r"alcoholic hepatitis|alcoholic cirrhosis|alcoholic liver"),

    ("peptic_ulcer",
     ["K25", "K26"],
     ["531", "532"],
     r"peptic ulcer|gastric ulcer|duodenal ulcer|\bPUD\b"),

    ("uti_pyelonephritis",
     ["N390", "N10"],
     ["5990"],
     r"urinary tract infection|pyelonephritis|\bUTI\b"),

    ("acute_kidney_injury",
     ["N179"],
     ["5849"],
     r"acute kidney|acute renal|\bAKI\b"),

    ("nephrolithiasis",
     ["N20"],
     ["592"],
     r"nephrolithiasis|kidney stone|renal calcul|renal stone|ureteral stone"),

    ("postop_complication",
     ["K9189", "T814"],
     ["99749", "99859"],
     r"postoperative|post-operative|surgical site|wound infection"),

    ("dyspepsia_reflux",
     ["K30", "K21"],
     ["5301"],
     r"dyspepsia|\bGERD\b|gastroesophageal reflux"),

    ("hepatitis_nonalcoholic",
     ["K758", "K729"],
     ["5728"],
     r"hepatitis"),

    ("chronic_pancreatitis",
     ["K861"],
     ["5771"],
     r"pancreatitis"),

    ("gi_malignancy",
     ["C786", "C18", "C25"],
     ["1970"],
     r"metastas|malignant|carcinoma|cancer|neoplasm"),

    ("hernia",
     ["K40", "K43"],
     ["550", "553"],
     r"hernia"),

    ("diverticulosis_noninflam",
     ["K5730", "K5750", "K5790"],
     ["56210"],
     r"diverticul"),

    ("mesenteric_ischemia",
     ["K550"],
     ["5570"],
     r"mesenteric ischemia|ischemic bowel|ischemic colitis"),

    ("gastroparesis",
     ["K3189"],
     ["53610"],
     r"gastroparesis"),
]


def build_group_icd_filter(v10_prefixes: list[str], v9_prefixes: list[str]) -> str:
    """Build SQL WHERE clause for ICD prefix matching."""
    parts = []
    for p in v10_prefixes:
        parts.append(f"(dx.icd_version = 10 AND dx.icd_code LIKE '{p}%')")
    for p in v9_prefixes:
        parts.append(f"(dx.icd_version = 9 AND dx.icd_code LIKE '{p}%')")
    return " OR ".join(parts)


def query_group(client: bigquery.Client, group_name: str,
                v10: list[str], v9: list[str],
                per_group: int, seed: int) -> list[int]:
    """Query BQ for hadm_ids matching one group, randomly sampled."""
    icd_filter = build_group_icd_filter(v10, v9)
    sql = f"""
    SELECT hadm_id FROM (
      SELECT DISTINCT dx.hadm_id
      FROM `{HOSP_DS}.diagnoses_icd` dx
      JOIN `{ED_DS}.edstays` es ON es.hadm_id = dx.hadm_id
      JOIN `{ED_DS}.triage` t ON t.stay_id = es.stay_id
      JOIN `{NOTE_DS}.discharge` n ON n.hadm_id = dx.hadm_id
      WHERE dx.seq_num = 1
        AND LOWER(t.chiefcomplaint) LIKE '%abd%'
        AND LOWER(t.chiefcomplaint) LIKE '%pain%'
        AND NOT ({EXCLUDE_TARGET})
        AND ({icd_filter})
    )
    ORDER BY FARM_FINGERPRINT(CAST(hadm_id AS STRING) || '{seed}')
    LIMIT {per_group}
    """
    df = client.query(sql).to_dataframe()
    return df["hadm_id"].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Create diverse 'other abdominal' cohort from 25 groups")
    parser.add_argument("-o", "--output", type=str,
                        default="bayes/other_abd_ids.txt")
    parser.add_argument("--per-group", type=int, default=30,
                        help="Hadm_ids to sample per group (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show groups and counts without sampling")
    parser.add_argument("--groups-output", type=str,
                        default="bayes/other_abd_groups.csv",
                        help="Save group assignments (hadm_id,group,hpi_pattern)")
    args = parser.parse_args()

    client = bigquery.Client()
    all_ids = []
    group_rows = []

    for name, v10, v9, hpi_pattern in GROUPS:
        if args.dry_run:
            # Just count available
            icd_filter = build_group_icd_filter(v10, v9)
            sql = f"""
            SELECT COUNT(DISTINCT dx.hadm_id) as n
            FROM `{HOSP_DS}.diagnoses_icd` dx
            JOIN `{ED_DS}.edstays` es ON es.hadm_id = dx.hadm_id
            JOIN `{ED_DS}.triage` t ON t.stay_id = es.stay_id
            JOIN `{NOTE_DS}.discharge` n ON n.hadm_id = dx.hadm_id
            WHERE dx.seq_num = 1
              AND LOWER(t.chiefcomplaint) LIKE '%abd%'
              AND LOWER(t.chiefcomplaint) LIKE '%pain%'
              AND NOT ({EXCLUDE_TARGET})
              AND ({icd_filter})
            """
            df = client.query(sql).to_dataframe()
            n = int(df.iloc[0, 0])
            print(f"  {name:35s} pool={n:5d}")
        else:
            ids = query_group(client, name, v10, v9, args.per_group, args.seed)
            print(f"  {name:35s} sampled={len(ids)}")
            for hid in ids:
                group_rows.append({"hadm_id": hid, "group": name,
                                   "hpi_exclude": hpi_pattern})
            all_ids.extend(ids)

    if args.dry_run:
        print(f"\n[dry-run] {len(GROUPS)} groups defined")
        return

    # Deduplicate (a hadm_id could theoretically match multiple groups)
    seen = set()
    unique_ids = []
    unique_rows = []
    for hid, row in zip(all_ids, group_rows):
        if hid not in seen:
            seen.add(hid)
            unique_ids.append(hid)
            unique_rows.append(row)

    # Save hadm_id list
    with open(args.output, "w") as f:
        for hid in sorted(unique_ids):
            f.write(f"{hid}\n")

    # Save group assignments
    import csv
    with open(args.groups_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hadm_id", "group", "hpi_exclude"])
        writer.writeheader()
        for row in sorted(unique_rows, key=lambda r: r["hadm_id"]):
            writer.writerow(row)

    print(f"\nTotal: {len(unique_ids)} hadm_ids ({len(all_ids) - len(unique_ids)} duplicates removed)")
    print(f"Saved to: {args.output}")
    print(f"Groups:   {args.groups_output}")


if __name__ == "__main__":
    main()
