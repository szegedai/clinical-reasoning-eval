"""Analyze replay results and produce visualizations + Excel summary.

Usage:
    # Single pathology
    python analyze_results.py results/appendicitis_gemini --pathology appendicitis

    # Multiple pathologies (multi-panel plots)
    python analyze_results.py results/appendicitis_gemini results/cholecystitis_gemini \
        --pathology appendicitis cholecystitis

    # Override target diagnosis name (if different from pathology key)
    python analyze_results.py results/diverticulitis_gemini --pathology diverticulitis \
        --target-dx diverticulitis

    # All options
    python analyze_results.py dir1 dir2 dir3 --pathology p1 p2 p3 \
        --output results/ --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dx_matcher import DxMatcher

MATCHER = DxMatcher()


def load_results(result_dir: Path) -> list[dict]:
    """Load all patient JSON files from a result directory."""
    files = sorted(result_dir.glob("patient_*.json"))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def entropy(confidences: list[float]) -> float:
    """Shannon entropy of a confidence distribution (bits)."""
    h = 0.0
    for p in confidences:
        if p > 0:
            h -= p * np.log2(p)
    return h


def extract_step_data(results: list[dict], pathology: str,
                      target_dx: str | None, max_steps: int) -> pd.DataFrame:
    """Extract per-patient per-step metrics into a DataFrame."""
    rows = []
    for patient in results:
        hadm_id = patient["hadm_id"]
        first_confident_step = patient.get("first_confident_step")
        for step in patient["steps"]:
            step_num = step["step"]
            if step_num > max_steps:
                break
            # Exclude post-confidence buffer steps (kept in JSON for case-by-case review)
            if first_confident_step is not None and step_num > first_confident_step:
                break
            parsed = step.get("parsed") or {}
            diff = parsed.get("differential") or []

            top1_dx = diff[0]["diagnosis"] if len(diff) > 0 else ""
            top1_conf = diff[0]["confidence"] if len(diff) > 0 else 0.0
            top1_correct = MATCHER.is_correct(top1_dx, pathology) if top1_dx else False
            top1_gracious = (
                top1_correct or MATCHER.is_gracious(top1_dx, pathology)
            ) if top1_dx else False

            top3_correct = any(
                MATCHER.is_correct(d["diagnosis"], pathology)
                for d in diff[:3]
            ) if diff else False
            top3_gracious = any(
                MATCHER.is_correct(d["diagnosis"], pathology)
                or MATCHER.is_gracious(d["diagnosis"], pathology)
                for d in diff[:3]
            ) if diff else False

            # Check if target ever appeared with >=90% confidence in any rank
            high_conf = any(
                MATCHER.is_correct(d["diagnosis"], pathology) and d["confidence"] >= 0.9
                for d in diff
            ) if diff else False

            # Confidence distribution entropy
            confs = [d["confidence"] for d in diff] if diff else []
            step_entropy = entropy(confs) if confs else None

            # New schema fields
            delta = parsed.get("delta")
            confident = parsed.get("confident_in_diagnosis")

            rows.append({
                "hadm_id": hadm_id,
                "step": step_num,
                "label": step.get("label", ""),
                "n_events": step.get("n_events", 0),
                "top1_dx": top1_dx,
                "top1_conf": top1_conf,
                "top1_correct": top1_correct,
                "top1_gracious": top1_gracious,
                "top3_correct": top3_correct,
                "top3_gracious": top3_gracious,
                "high_conf_correct": high_conf,
                "ddx_size": len(diff),
                "entropy": step_entropy,
                "delta": delta,
                "confident_in_diagnosis": confident,
                "first_confident_step": first_confident_step,
                "input_tokens": step.get("input_tokens", 0),
                "output_tokens": step.get("output_tokens", 0),
                "latency_ms": step.get("latency_ms", 0),
                "assessment": parsed.get("assessment", ""),
            })
    return pd.DataFrame(rows)


def compute_accuracy_series(df: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """Compute per-step accuracy metrics across patients."""
    records = []

    for step_num in range(1, max_steps + 1):
        step_df = df[df["step"] == step_num]
        active = len(step_df)
        if active == 0:
            continue

        top1_acc = step_df["top1_correct"].mean()
        top1_gracious_acc = step_df["top1_gracious"].mean()
        top3_acc = step_df["top3_correct"].mean()
        top3_gracious_acc = step_df["top3_gracious"].mean()

        # Cumulative "ever correct" up to this step
        up_to = df[df["step"] <= step_num]
        ever_top1 = up_to.groupby("hadm_id")["top1_correct"].any().mean()
        ever_high_conf = up_to.groupby("hadm_id")["high_conf_correct"].any().mean()

        # Entropy and confidence signal
        mean_entropy = step_df["entropy"].mean() if step_df["entropy"].notna().any() else None
        pct_confident = step_df["confident_in_diagnosis"].mean() if step_df["confident_in_diagnosis"].notna().any() else None

        records.append({
            "step": step_num,
            "top1_accuracy": top1_acc,
            "top1_gracious": top1_gracious_acc,
            "top3_accuracy": top3_acc,
            "top3_gracious": top3_gracious_acc,
            "ever_top1": ever_top1,
            "ever_high_conf": ever_high_conf,
            "mean_entropy": mean_entropy,
            "pct_confident": pct_confident,
            "active_patients": active,
        })
    return pd.DataFrame(records)


def plot_accuracy(all_acc: dict[str, pd.DataFrame], all_counts: dict[str, int],
                  model_name: str, output_path: Path, max_steps: int):
    """Plot accuracy over steps, one subplot per pathology."""
    n = len(all_acc)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Diagnostic accuracy over replay steps — {model_name}", fontsize=13, y=0.98)

    for idx, (pathology, acc) in enumerate(all_acc.items()):
        ax = axes[idx // cols][idx % cols]
        ax2 = ax.twinx()

        n_patients = all_counts[pathology]
        final_top1 = acc["top1_accuracy"].iloc[-1] if len(acc) > 0 else 0
        ever_high = acc["ever_high_conf"].iloc[-1] if len(acc) > 0 else 0

        # Bar chart for active patients
        ax2.bar(acc["step"], acc["active_patients"], alpha=0.15, color="gray", zorder=0)
        ax2.set_ylabel("patients active", fontsize=9, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)

        # Lines
        ax.plot(acc["step"], acc["top1_accuracy"], "o-", color="C0", ms=4, label="top-1 accuracy")
        ax.plot(acc["step"], acc["top3_accuracy"], "s-", color="C1", ms=4, label="top-3 accuracy")
        ax.plot(acc["step"], acc["ever_top1"], "^--", color="C2", ms=4,
                label="ever matched #1 (cumulative)")
        ax.plot(acc["step"], acc["ever_high_conf"], "d--", color="C3", ms=4,
                label="ever >=90% confidence (cumulative)")

        label = pathology.replace("_", " ")
        ax.set_title(f"{label} (n={n_patients})\n"
                     f"final top-1: {final_top1:.0%} | ever >=90% conf: {ever_high:.0%}",
                     fontsize=10)
        ax.set_xlabel("replay step", fontsize=9)
        ax.set_ylabel("accuracy / ratio", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.5, max_steps + 0.5)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Single legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_confidence_trajectories(all_step_data: dict[str, pd.DataFrame],
                                 all_counts: dict[str, int],
                                 model_name: str, output_path: Path, max_steps: int):
    """Plot per-patient top-1 confidence trajectories, colored by correctness."""
    n = len(all_step_data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Top-1 confidence trajectory per patient — {model_name}", fontsize=13, y=0.98)

    for idx, (pathology, df) in enumerate(all_step_data.items()):
        ax = axes[idx // cols][idx % cols]
        n_patients = all_counts[pathology]

        for hadm_id, pdf in df.groupby("hadm_id"):
            steps = pdf["step"].values
            confs = pdf["top1_conf"].values
            correct = pdf["top1_correct"].values

            # Color: green if mostly correct, red if mostly wrong
            frac_correct = correct.mean()
            if frac_correct > 0.5:
                color = "green"
                style = "-"
            else:
                color = "red"
                style = "--"

            ax.plot(steps, confs, style, color=color, alpha=0.2, linewidth=0.8)

        # Median lines
        median_all = df.groupby("step")["top1_conf"].median()
        correct_df = df[df["top1_correct"]]
        incorrect_df = df[~df["top1_correct"]]

        ax.plot(median_all.index, median_all.values, "k-", linewidth=2.5, label="median (all)")

        if len(correct_df) > 0:
            med_correct = correct_df.groupby("step")["top1_conf"].median()
            ax.plot(med_correct.index, med_correct.values, "g--", linewidth=2,
                    label="median (correct)")

        if len(incorrect_df) > 0:
            med_incorrect = incorrect_df.groupby("step")["top1_conf"].median()
            ax.plot(med_incorrect.index, med_incorrect.values, "r--", linewidth=2,
                    label="median (incorrect)")

        label = pathology.replace("_", " ")
        ax.set_title(f"{label} (n={n_patients})", fontsize=10)
        ax.set_xlabel("replay step", fontsize=9)
        ax.set_ylabel("top-1 confidence", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.5, max_steps + 0.5)
        ax.grid(alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Legend from first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Add individual line legend entries
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], color="green", alpha=0.5, label="correct top-1"),
        Line2D([0], [0], color="red", linestyle="--", alpha=0.5, label="incorrect top-1"),
    ]
    labels += ["correct top-1", "incorrect top-1"]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_entropy_and_confidence(all_acc: dict[str, pd.DataFrame], all_counts: dict[str, int],
                                model_name: str, output_path: Path, max_steps: int):
    """Plot entropy and confidence signal over steps, with active patient count and top-1 accuracy."""
    n = len(all_acc)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Diagnostic certainty over replay steps — {model_name}", fontsize=13, y=0.98)

    for idx, (pathology, acc) in enumerate(all_acc.items()):
        ax = axes[idx // cols][idx % cols]
        n_patients = all_counts[pathology]

        # Active patients as background bars
        ax2 = ax.twinx()
        ax2.bar(acc["step"], acc["active_patients"], alpha=0.10, color="gray", zorder=0)
        ax2.set_ylabel("patients active", fontsize=9, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)

        # Entropy on left axis
        if acc["mean_entropy"].notna().any():
            ax.plot(acc["step"], acc["mean_entropy"], "o-", color="C0", ms=4, label="mean entropy (bits)")
        ax.set_ylabel("entropy (bits) / fraction", fontsize=9)
        ax.set_ylim(bottom=0)

        # % confident
        if acc["pct_confident"].notna().any():
            ax.plot(acc["step"], acc["pct_confident"], "s-", color="C3", ms=4, label="% confident")
            ax.fill_between(acc["step"], 0, acc["pct_confident"], alpha=0.1, color="C3")

        # Top-1 accuracy overlay
        ax.plot(acc["step"], acc["top1_accuracy"], "^--", color="C2", ms=4, alpha=0.7, label="top-1 accuracy")

        label = pathology.replace("_", " ")
        ax.set_title(f"{label} (n={n_patients})", fontsize=10)
        ax.set_xlabel("replay step", fontsize=9)
        ax.set_xlim(0.5, max_steps + 0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)



def classify_termination(patient: dict, max_steps: int) -> str:
    """Classify why a patient's replay ended."""
    fc = patient.get("first_confident_step")
    steps = patient.get("steps", [])
    if not steps:
        return "no_steps"
    last_step = steps[-1]
    last_step_num = last_step["step"]

    if fc is not None:
        return "confident"
    if last_step_num >= max_steps:
        return "max_steps"
    # Check for surgery stop (Service: SURG in last step's prompt)
    if "Service: SURG" in last_step.get("prompt", ""):
        return "surgery"
    return "timeline_end"


def compute_termination_accuracy(results: list[dict], pathology: str,
                                 max_steps: int) -> pd.DataFrame:
    """Compute accuracy at termination step, grouped by termination reason."""
    rows = []
    for patient in results:
        reason = classify_termination(patient, max_steps)
        fc = patient.get("first_confident_step")
        steps = patient.get("steps", [])

        # Pick the termination step to evaluate
        if reason == "confident" and fc is not None:
            # Evaluate at the confidence step
            eval_steps = [s for s in steps if s["step"] == fc]
        else:
            # Evaluate at the last step
            eval_steps = [steps[-1]] if steps else []

        if not eval_steps:
            continue

        step = eval_steps[0]
        parsed = step.get("parsed") or {}
        diff = parsed.get("differential") or []
        top1_dx = diff[0]["diagnosis"] if diff else ""

        rows.append({
            "hadm_id": patient["hadm_id"],
            "reason": reason,
            "eval_step": step["step"],
            "top1_dx": top1_dx,
            "top1_correct": MATCHER.is_correct(top1_dx, pathology) if top1_dx else False,
            "top1_gracious": (
                MATCHER.is_correct(top1_dx, pathology) or MATCHER.is_gracious(top1_dx, pathology)
            ) if top1_dx else False,
            "top3_correct": any(
                MATCHER.is_correct(d["diagnosis"], pathology) for d in diff[:3]
            ) if diff else False,
        })
    return pd.DataFrame(rows)


def plot_termination_accuracy(all_term: dict[str, pd.DataFrame], all_counts: dict[str, int],
                              model_name: str, output_path: Path):
    """Grouped bar chart: accuracy at termination by reason, per pathology."""
    reason_order = ["confident", "surgery", "timeline_end", "max_steps"]
    reason_labels = {
        "confident": "confident",
        "surgery": "surgery stop",
        "timeline_end": "timeline end",
        "max_steps": "max steps",
    }
    reason_colors = {
        "confident": "#2ecc71",
        "surgery": "#3498db",
        "timeline_end": "#95a5a6",
        "max_steps": "#e74c3c",
    }

    pathologies = list(all_term.keys())
    n_path = len(pathologies)

    fig, ax = plt.subplots(figsize=(max(10, n_path * 1.8), 6))
    fig.suptitle(f"Top-1 accuracy at termination — {model_name}", fontsize=13)

    bar_width = 0.18
    x = np.arange(n_path)

    for i, reason in enumerate(reason_order):
        accs = []
        counts = []
        for pathology in pathologies:
            df = all_term[pathology]
            subset = df[df["reason"] == reason]
            if len(subset) > 0:
                accs.append(subset["top1_correct"].mean())
                counts.append(len(subset))
            else:
                accs.append(0)
                counts.append(0)

        offset = (i - len(reason_order) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accs, bar_width,
                      label=reason_labels[reason], color=reason_colors[reason],
                      edgecolor="white", linewidth=0.5)

        # Count labels on bars
        for j, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        str(count), ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in pathologies], fontsize=9)
    ax.set_ylabel("top-1 accuracy", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, title="termination reason")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def print_termination_table(all_term: dict[str, pd.DataFrame]):
    """Print a summary table of accuracy by termination reason."""
    reason_order = ["confident", "surgery", "timeline_end", "max_steps"]

    header = "{:28s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "", "confident", "surgery", "timeline end", "max steps")
    print("\n" + header)
    print("-" * len(header))

    for pathology, df in all_term.items():
        parts = []
        for reason in reason_order:
            subset = df[df["reason"] == reason]
            if len(subset) > 0:
                acc = subset["top1_correct"].mean()
                parts.append("{:.0%} ({})".format(acc, len(subset)))
            else:
                parts.append("—")
        name = pathology.replace("_", " ")
        print("{:28s} {:>12s} {:>12s} {:>12s} {:>12s}".format(name, *parts))

    # Overall row
    all_df = pd.concat(all_term.values())
    parts = []
    for reason in reason_order:
        subset = all_df[all_df["reason"] == reason]
        if len(subset) > 0:
            acc = subset["top1_correct"].mean()
            parts.append("{:.0%} ({})".format(acc, len(subset)))
        else:
            parts.append("—")
    print("-" * len(header))
    print("{:28s} {:>12s} {:>12s} {:>12s} {:>12s}".format("OVERALL", *parts))
    print()


def save_excel(all_step_data: dict[str, pd.DataFrame], output_path: Path):
    """Save per-step summary to Excel, one sheet per pathology."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for pathology, df in all_step_data.items():
            # Per-patient summary: one row per patient with step-level columns
            summary = df.pivot_table(
                index="hadm_id",
                columns="step",
                values=["top1_dx", "top1_conf", "top1_correct", "delta",
                        "confident_in_diagnosis", "entropy", "assessment"],
                aggfunc="first",
            )
            # Flatten column names
            summary.columns = [f"{col}_{step}" for col, step in summary.columns]
            summary.to_excel(writer, sheet_name=pathology[:31])
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze replay results and produce visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("result_dirs", nargs="+", help="Result directories (one per pathology)")
    parser.add_argument("--pathology", nargs="+", required=True,
                        help="Pathology name for each result dir (same order)")
    parser.add_argument("--target-dx", nargs="*", default=None,
                        help="Override target diagnosis name(s)")
    parser.add_argument("--output", "-o", type=str, default="results",
                        help="Output directory for plots and excel (default: results/)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps to plot (default: 20)")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel output")
    args = parser.parse_args()

    if len(args.result_dirs) != len(args.pathology):
        sys.exit("Number of result_dirs must match number of --pathology args")

    target_dxs = args.target_dx or [None] * len(args.pathology)
    if len(target_dxs) != len(args.pathology):
        sys.exit("Number of --target-dx must match number of --pathology args")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect model name from first run_config
    model_name = "unknown"
    rc_path = Path(args.result_dirs[0]) / "run_config.json"
    if rc_path.exists():
        with open(rc_path) as f:
            model_name = json.load(f).get("model", model_name)

    all_step_data = {}
    all_acc = {}
    all_counts = {}
    all_raw = {}

    for rdir, pathology, tdx in zip(args.result_dirs, args.pathology, target_dxs):
        rdir = Path(rdir)
        print(f"Loading {pathology} from {rdir}...")
        results = load_results(rdir)
        if not results:
            print(f"  No patient files found, skipping.")
            continue

        df = extract_step_data(results, pathology, tdx, args.max_steps)
        acc = compute_accuracy_series(df, args.max_steps)

        all_step_data[pathology] = df
        all_acc[pathology] = acc
        all_counts[pathology] = len(results)
        all_raw[pathology] = results

        print(f"  {len(results)} patients, {len(df)} step records")

    if not all_step_data:
        sys.exit("No data loaded.")

    # Plots
    plot_accuracy(all_acc, all_counts, model_name, out_dir / "accuracy_over_steps.png",
                  args.max_steps)
    plot_confidence_trajectories(all_step_data, all_counts, model_name,
                                out_dir / "confidence_trajectories_all.png", args.max_steps)
    plot_entropy_and_confidence(all_acc, all_counts, model_name,
                               out_dir / "entropy_and_confidence.png", args.max_steps)
    # Termination accuracy
    all_term = {}
    for pathology, results in all_raw.items():
        all_term[pathology] = compute_termination_accuracy(results, pathology, args.max_steps)
    plot_termination_accuracy(all_term, all_counts, model_name,
                              out_dir / "termination_accuracy.png")
    print_termination_table(all_term)

    # Excel
    if not args.no_excel:
        for pathology, df in all_step_data.items():
            excel_path = out_dir / f"{pathology}_results.xlsx"
            save_excel({pathology: df}, excel_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
