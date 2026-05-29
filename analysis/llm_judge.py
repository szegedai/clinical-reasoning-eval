"""LLM-as-a-judge for diagnosis matching.

Collects regex non-hits from replay results and sends them through an LLM
to evaluate whether the model's diagnosis is clinically equivalent to the
gold-standard label.

Usage:
    # Collect non-hits and judge them
    python -m clinical-reasoning-eval.analysis.llm_judge \
        results_2026-03-28/appendicitis_gemini \
        results_2026-03-28/cholecystitis_gemini \
        --pathology appendicitis cholecystitis \
        -o results_2026-03-28/judge_results.json

    # Dry run: collect non-hits without calling the LLM
    python -m clinical-reasoning-eval.analysis.llm_judge \
        results_2026-03-28/appendicitis_gemini \
        --pathology appendicitis --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.dx_matcher import DxMatcher

MATCHER = DxMatcher()

JUDGE_SYSTEM_PROMPT = """\
You are a medical evaluation judge. Your task is to decide whether a \
clinician's working diagnosis is clinically equivalent to or a reasonable \
match for a known gold-standard diagnosis.

This is NOT about whether the clinician's diagnosis is the single best \
possible answer. It IS about whether the two diagnoses refer to the same \
underlying clinical condition, even if described at a different level of \
specificity or from a different clinical angle.

Guidelines:
- A match means the clinician has essentially identified the correct \
condition, even if they used different terminology, focused on a \
complication of the same disease, or named a broader/narrower category.
- A non-match means the clinician identified a genuinely different \
condition, even if it shares some symptoms or anatomy.

Examples of MATCHES (score=1):
- Gold: "Acute cholecystitis" → Clinician: "Biliary colic" (same organ, same disease spectrum)
- Gold: "Diverticulitis" → Clinician: "Perforated sigmoid" (complication of the same condition)
- Gold: "Acute appendicitis" → Clinician: "Ruptured appendicitis with peritonitis" (same disease, advanced stage)
- Gold: "TIA" → Clinician: "Acute ischemic stroke" (same vascular mechanism, TIA/stroke is a spectrum)

Examples of NON-MATCHES (score=0):
- Gold: "Acute appendicitis" → Clinician: "Acute cholecystitis" (different organ entirely)
- Gold: "TIA" → Clinician: "Migraine with aura" (different mechanism)
- Gold: "Cholecystitis" → Clinician: "Peptic ulcer disease" (different condition despite RUQ pain)
- Gold: "Diverticulitis" → Clinician: "Small bowel obstruction" (different pathology)
"""

JUDGE_USER_TEMPLATE = """\
Gold-standard diagnosis: {gold}
Clinician's working diagnosis: {prediction}

Is the clinician's diagnosis a clinical match for the gold standard?

Respond with a JSON object:
```json
{{"score": <0 or 1>, "reasoning": "<1-2 sentences explaining your judgement>"}}
```"""


def collect_nonhits(result_dir: Path, pathology: str) -> list[dict]:
    """Collect all unique LLM diagnoses that don't match regex patterns."""
    files = sorted(result_dir.glob("patient_*.json"))
    nonhits = {}  # (diagnosis, pathology) -> example info

    for f in files:
        data = json.load(open(f))
        hadm_id = data["hadm_id"]
        first_confident_step = data.get("first_confident_step")

        for step in data["steps"]:
            step_num = step["step"]
            # Skip post-confidence buffer steps
            if first_confident_step is not None and step_num > first_confident_step:
                break

            parsed = step.get("parsed") or {}
            diff = parsed.get("differential") or []

            for rank, entry in enumerate(diff):
                dx = entry.get("diagnosis", "")
                if not dx:
                    continue
                is_match = MATCHER.is_correct(dx, pathology)
                is_gracious = MATCHER.is_gracious(dx, pathology)
                if not is_match and not is_gracious:
                    key = dx.strip().lower()
                    if key not in nonhits:
                        nonhits[key] = {
                            "diagnosis": dx,
                            "pathology": pathology,
                            "example_hadm_id": hadm_id,
                            "example_step": step_num,
                            "example_rank": rank + 1,
                            "count": 0,
                        }
                    nonhits[key]["count"] += 1

    return sorted(nonhits.values(), key=lambda x: -x["count"])


def judge_batch(nonhits: list[dict], client: openai.OpenAI, model: str,
                gold_label: str) -> list[dict]:
    """Send non-hits to the LLM judge and collect scores."""
    results = []
    for i, item in enumerate(nonhits):
        prompt = JUDGE_USER_TEMPLATE.format(
            gold=gold_label,
            prediction=item["diagnosis"],
        )

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""

                # Parse JSON from response
                parsed = None
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    import re
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except json.JSONDecodeError:
                            pass

                result = {
                    **item,
                    "gold_label": gold_label,
                    "score": parsed.get("score") if parsed else None,
                    "reasoning": parsed.get("reasoning", "") if parsed else text,
                    "raw_response": text,
                }
                results.append(result)

                status = "match" if result["score"] == 1 else "no match" if result["score"] == 0 else "???"
                print(f"  [{i+1}/{len(nonhits)}] {item['diagnosis'][:50]:50s} → {status} (n={item['count']})")
                break

            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    results.append({
                        **item,
                        "gold_label": gold_label,
                        "score": None,
                        "reasoning": f"API error: {e}",
                        "raw_response": "",
                    })
                    print(f"  [{i+1}/{len(nonhits)}] {item['diagnosis'][:50]:50s} → FAILED")

    return results


def print_summary(all_results: list[dict]):
    """Print summary of judge results."""
    by_pathology = {}
    for r in all_results:
        p = r["pathology"]
        if p not in by_pathology:
            by_pathology[p] = {"matches": [], "nonmatches": [], "errors": []}
        if r["score"] == 1:
            by_pathology[p]["matches"].append(r)
        elif r["score"] == 0:
            by_pathology[p]["nonmatches"].append(r)
        else:
            by_pathology[p]["errors"].append(r)

    print("\n" + "=" * 80)
    print("LLM JUDGE SUMMARY")
    print("=" * 80)

    for pathology, groups in by_pathology.items():
        n_match = len(groups["matches"])
        n_nomatch = len(groups["nonmatches"])
        n_err = len(groups["errors"])
        total = n_match + n_nomatch + n_err
        match_occ = sum(r["count"] for r in groups["matches"])
        nomatch_occ = sum(r["count"] for r in groups["nonmatches"])

        print(f"\n{pathology}:")
        print(f"  {total} unique non-hit diagnoses judged")
        print(f"  {n_match} matches ({match_occ} occurrences)")
        print(f"  {n_nomatch} non-matches ({nomatch_occ} occurrences)")
        if n_err:
            print(f"  {n_err} errors")

        if groups["matches"]:
            print(f"\n  Matches (would recover):")
            for r in sorted(groups["matches"], key=lambda x: -x["count"])[:10]:
                print(f"    {r['count']:4d}x  {r['diagnosis'][:60]}")
                print(f"           {r['reasoning']}")

        if groups["nonmatches"]:
            print(f"\n  Top non-matches:")
            for r in sorted(groups["nonmatches"], key=lambda x: -x["count"])[:10]:
                print(f"    {r['count']:4d}x  {r['diagnosis'][:60]}")
                print(f"           {r['reasoning']}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge for diagnosis matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("result_dirs", nargs="+", help="Result directories")
    parser.add_argument("--pathology", nargs="+", required=True,
                        help="Pathology name per result dir")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path for judge results")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="LLM model for judging (default: gemini-2.0-flash)")
    parser.add_argument("--base-url", type=str,
                        default="https://generativelanguage.googleapis.com/v1beta/openai/",
                        help="API base URL")
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY",
                        help="Environment variable for API key")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect non-hits without calling the LLM")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Only judge diagnoses with >= N occurrences (default: 1)")
    args = parser.parse_args()

    if len(args.result_dirs) != len(args.pathology):
        sys.exit("Number of result_dirs must match --pathology args")

    # Gold labels: human-readable pathology names
    gold_labels = {
        "appendicitis": "Acute appendicitis",
        "cholecystitis": "Acute cholecystitis",
        "diverticulitis": "Acute diverticulitis",
        "acute_pancreatitis": "Acute pancreatitis",
        "ischemic_stroke": "Acute ischemic stroke",
        "intracerebral_hemorrhage": "Intracerebral hemorrhage",
        "tia": "Transient ischemic attack (TIA)",
    }

    # Collect all non-hits
    all_nonhits = []
    for rdir, pathology in zip(args.result_dirs, args.pathology):
        rdir = Path(rdir)
        print(f"Collecting non-hits for {pathology} from {rdir}...")
        nonhits = collect_nonhits(rdir, pathology)
        nonhits = [n for n in nonhits if n["count"] >= args.min_count]
        print(f"  {len(nonhits)} unique diagnoses (>= {args.min_count} occurrences)")
        all_nonhits.extend(nonhits)

    print(f"\nTotal: {len(all_nonhits)} unique non-hit diagnoses to judge")

    if args.dry_run:
        for item in all_nonhits[:30]:
            print(f"  {item['count']:4d}x  [{item['pathology']}] {item['diagnosis']}")
        if len(all_nonhits) > 30:
            print(f"  ... and {len(all_nonhits) - 30} more")
        return

    # Set up LLM client
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        sys.exit(f"Set {args.api_key_env} environment variable")

    client = openai.OpenAI(base_url=args.base_url, api_key=api_key)

    # Judge each pathology
    all_results = []
    for pathology in dict.fromkeys(p for _, p in zip(args.result_dirs, args.pathology)):
        gold = gold_labels.get(pathology, pathology)
        nonhits = [n for n in all_nonhits if n["pathology"] == pathology]
        if not nonhits:
            continue
        print(f"\nJudging {pathology} ({len(nonhits)} diagnoses, gold: {gold})...")
        results = judge_batch(nonhits, client, args.model, gold)
        all_results.extend(results)

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_results)} judgements to {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
