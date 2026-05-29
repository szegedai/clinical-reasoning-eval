"""Filter 'other abdominal' cohort: remove patients whose HPI mentions their
diagnosis (per-group regex patterns from other_abd_groups.csv).

Usage:
    python bayes/filter_hpi_leaks.py --timelines-dir timelines_2026-03-28/other_abdominal --dry-run
    python bayes/filter_hpi_leaks.py --timelines-dir timelines_2026-03-28/other_abdominal
"""

import argparse
import csv
import re
from pathlib import Path


def extract_hpi_from_timeline(timeline_path: Path) -> str | None:
    """Extract HPI text from a timeline CSV (event_type == DISCHARGE_HPI)."""
    with open(timeline_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event_type") == "DISCHARGE_HPI":
                return row.get("description", "")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter other abdominal cohort by HPI leak detection")
    parser.add_argument("--timelines-dir", type=str, required=True)
    parser.add_argument("--groups-file", type=str,
                        default="bayes/other_abd_groups.csv")
    parser.add_argument("--output", type=str,
                        default="bayes/other_abd_ids_filtered.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timelines_dir = Path(args.timelines_dir)

    # Load group assignments with HPI patterns
    with open(args.groups_file) as f:
        reader = csv.DictReader(f)
        groups = list(reader)

    print(f"Input: {len(groups)} patients across groups")

    kept = []
    excluded = []
    no_timeline = []
    no_hpi = []

    for row in groups:
        hid = int(row["hadm_id"])
        pattern = row["hpi_exclude"]
        path = timelines_dir / f"timeline_{hid}.csv"

        if not path.exists():
            no_timeline.append(hid)
            continue

        hpi = extract_hpi_from_timeline(path)
        if hpi is None:
            no_hpi.append(hid)
            kept.append(row)
            continue

        if re.search(pattern, hpi, re.IGNORECASE):
            excluded.append((hid, row["group"], pattern))
        else:
            kept.append(row)

    print(f"\nResults:")
    print(f"  Kept:         {len(kept)}")
    print(f"  Excluded:     {len(excluded)} (HPI mentions diagnosis)")
    print(f"  No timeline:  {len(no_timeline)}")
    print(f"  No HPI event: {len(no_hpi)} (kept)")

    # Per-group summary
    from collections import Counter
    kept_groups = Counter(r["group"] for r in kept)
    excl_groups = Counter(g for _, g, _ in excluded)
    all_groups = sorted(set(list(kept_groups.keys()) + list(excl_groups.keys())))
    print(f"\nPer-group breakdown:")
    print(f"  {'Group':35s} {'Kept':>5s} {'Excl':>5s}")
    print(f"  {'-'*47}")
    for g in all_groups:
        print(f"  {g:35s} {kept_groups.get(g,0):5d} {excl_groups.get(g,0):5d}")

    if args.dry_run:
        if excluded:
            print(f"\nExcluded examples (first 10):")
            for hid, group, pat in excluded[:10]:
                print(f"  {hid}  {group:30s}  /{pat}/")
        print(f"\n[dry-run] Would save {len(kept)} hadm_ids to {args.output}")
        return

    kept_ids = sorted(int(r["hadm_id"]) for r in kept)
    with open(args.output, "w") as f:
        for hid in kept_ids:
            f.write(f"{hid}\n")
    print(f"\nSaved {len(kept)} hadm_ids to {args.output}")


if __name__ == "__main__":
    main()
