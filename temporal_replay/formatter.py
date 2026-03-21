from __future__ import annotations

import pandas as pd


def _ordinal(n: int) -> str:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th"
    )
    return f"{n}{suffix}"


def _fmt_time(event_time) -> str:
    """Format as 'March 14th at 09:30'. Year is omitted (MIMIC dates are shifted)."""
    dt = pd.Timestamp(event_time)
    return f"{dt.strftime('%B')} {_ordinal(dt.day)} at {dt.strftime('%H:%M')}"


# {time}, {elapsed_h}, {description}, {value}, {unit}, {flag}
DEFAULT_TEMPLATES: dict[str, str] = {
    # ED flow
    "ED_ARRIVAL":           "At {time}, {description}.",
    "ED_DEPARTURE":         "At {time}: {description}.",
    "ADMISSION":            "At {time}: {description}.",
    "DISCHARGE":            "At {time}: {description}.",
    "TRANSFER":             "At {time}: {description}.",
    "SERVICE":              "At {time}: {description}.",
    # Triage
    "TRIAGE_COMPLAINT":     "Triage complaint: {description}.",
    "TRIAGE_PAIN":          "Triage pain score: {description}.",
    "TRIAGE_ACUITY":        "Triage acuity (ESI level {value}).",
    "TRIAGE_VITAL":         "Triage vital — {description}.",
    # Vitals & medications
    "ED_VITALS":            "Vitals at {time}: {description}.",
    "ED_PYXIS":             "ED medication dispensed ({time}): {description}.",
    "MED_ADMIN":            "At {time}: {description}.",
    "RX_START":             "Prescription started at {time}: {description}.",
    # Diagnostics & labs
    "ED_DIAGNOSIS":         "{description}.",
    "IMAGING_STUDY":        "Imaging ordered at {time}: {description}.",
    "RADIOLOGY_REPORT":     "Radiology report:\n{description}",
    "SPECIMEN_COLLECTED":   "Lab specimen collected at {time}: {description}.",
    "LAB_RESULT":           "Lab result ({time}): {description}.",
    "MICRO_SAMPLE":         "Microbiology sample collected at {time}: {description}.",
    "MICRO_RESULT":         "Microbiology result ({time}): {description}.",
    # Procedures & medications
    "PROCEDURE_ICD":        "Procedure at {time}: {description}.",
    "MED_RECON":            "Medication reconciliation: {description}.",
    # ICU
    "ICU_VITAL":            "ICU vital ({time}): {description}.",
    "ICU_INPUT":            "ICU fluid input ({time}): {description}.",
    "ICU_OUTPUT":           "ICU fluid output ({time}): {description}.",
    "ICU_PROCEDURE":        "ICU procedure ({time}): {description}.",
    # Discharge note sections
    "DISCHARGE_HPI":        "History of present illness:\n{description}",
    "DISCHARGE_PE":         "Physical exam:\n{description}",
    "DISCHARGE_DX":         "{description}.",
    "DISCHARGE_FREETEXTDX": "Discharge diagnosis: {description}.",
    "DISCHARGE_NOTE":       "Discharge note:\n{description}",
}

_FALLBACK_TEMPLATE = "At {time}: {description}."


class PromptFormatter:
    """Format timeline events into human-readable strings for LLM prompts.

    Usage:
      fmt = PromptFormatter()
      lines = fmt.format_events(chunk.events)
      lines = fmt.format_events(chunk.cumulative)
    """

    def __init__(self):
        self.templates = DEFAULT_TEMPLATES

    def format_events(self, df: pd.DataFrame) -> list[str]:
        """Format each row as a string. Returns one entry per event."""
        results = []
        for _, row in df.iterrows():
            template = self.templates.get(row["event_type"], _FALLBACK_TEMPLATE)
            ctx = {
                "time":        _fmt_time(row["event_time"]),
                "elapsed_h":   row["elapsed_hours"],
                "description": row["description"] if pd.notna(row.get("description")) else "",
                "value":       row["value"] if pd.notna(row.get("value")) else "",
                "unit":        row["unit"] if pd.notna(row.get("unit")) else "",
                "flag":        row["flag"] if pd.notna(row.get("flag")) else "",
            }
            results.append(template.format(**ctx))
        return results

    def format_events_numbered(self, df: pd.DataFrame, start_index: int = 0) -> list[str]:
        """Format events with global cumulative indices: '[N] event text'."""
        lines = self.format_events(df)
        return [f"[{start_index + i}] {line}" for i, line in enumerate(lines)]
