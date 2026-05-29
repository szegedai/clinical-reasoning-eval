"""Central configuration for bias-resilience experiments.

Paths, model configs, condition registry, and run-level constants.
Patient data paths are resolved relative to MIMIC_ROOT (outside the repo).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]   # clinical-reasoning-eval-phase1/
REPO_ROOT = _REPO_ROOT                             # public alias for cli.py
MIMIC_ROOT = _REPO_ROOT.parent                      # mimic_data/

load_dotenv(MIMIC_ROOT / ".env")

# Default timeline root — override via --timelines-root CLI flag or env var
TIMELINES_ROOT = Path(
    os.environ.get("BIAS_TIMELINES_ROOT", str(MIMIC_ROOT / "timelines_2026-03-28"))
)

# Run outputs go here (outside the repo, patient-data side)
RESULTS_ROOT = Path(
    os.environ.get("BIAS_RESULTS_ROOT", str(MIMIC_ROOT / "bias_resilience_runs"))
)

SCHEMA_VERSION = 1
PROMPT_VERSION = "v1"

# Human-readable correct-dx label per pathology, used in struc_belief WD prose.
# Matches demote_correct.py's PATHOLOGY_LABEL but properly capitalised for clinical prose.
PATHOLOGY_CORRECT_LABEL: dict[str, str] = {
    "appendicitis":            "Acute appendicitis",
    "cholecystitis":           "Cholecystitis",
    "diverticulitis":          "Diverticulitis",
    "acute_pancreatitis":      "Acute pancreatitis",
    "ischemic_stroke":         "Ischemic stroke",
    "intracerebral_hemorrhage": "Intracerebral hemorrhage",
    "tia":                     "Transient ischemic attack",
    "other_abdominal":         "Other abdominal pathology",
    "other_abdominal_no_hpi":  "Other abdominal pathology",
}
# Pinned cohort manifests: run-id → path relative to REPO_ROOT.
# When the CLI run-id matches a key here and --cohort-file is not set,
# the manifest is used automatically (prevents file-system-order drift).
DEFAULT_COHORTS: dict[str, str] = {
    "phase1_gemma4_cholecystitis":       "bias_resilience/cohorts/phase1_n50_cholecystitis.txt",
    "phase1_gemma4_appendicitis":        "bias_resilience/cohorts/phase1_n50_appendicitis.txt",
    "phase1_gemma4_acute_pancreatitis":  "bias_resilience/cohorts/phase1_n50_acute_pancreatitis.txt",
    "phase1_gemma4_diverticulitis":      "bias_resilience/cohorts/phase1_n50_diverticulitis.txt",
}

# Canonical chunker kwargs applied to every replay and anchor resolution.
# Ported from trial_v2/run_trial.py — omitted from orchestrators by oversight.
DEFAULT_CHUNKER_KWARGS: dict = {
    "exclude_sources": {"ICU"},
    "exclude_event_types": {"DISCHARGE_DX", "DISCHARGE_FREETEXTDX", "ED_DIAGNOSIS"},
    "stop_at": {"event_type": "SERVICE", "description": "Service: SURG"},
}

MAX_RETRIES = 3
CALL_TIMEOUT_S = 120
DEFAULT_WORKERS = 16   # --workers N; crank to 128 for large remote batches


@dataclass
class ModelConfig:
    """Configuration for one (provider, model) pair."""
    name: str                    # human-readable key used in run paths
    model_id: str                # provider-side model identifier
    base_url: str
    api_key_env: str = ""        # name of the env var holding the key; empty = no auth required
    temperature: float = 0.0
    max_tokens: int = 1024
    extra_headers: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)
    # Pricing (USD per 1 M tokens) — used only for pre-flight cost estimate
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    thinking_price_per_1m: float = 0.0

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return "local"
        v = os.environ.get(self.api_key_env, "")
        if not v:
            raise RuntimeError(
                f"API key env var '{self.api_key_env}' is not set. "
                "Check your .env file."
            )
        return v


MODELS: dict[str, ModelConfig] = {
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        model_id="google/gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        temperature=0.0,
        max_tokens=1024,
        # Disable thinking tokens; still captured in tokens_thinking if surfaced.
        extra_body={"google": {"thinkingConfig": {"thinkingBudget": 0}}},
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
        thinking_price_per_1m=3.50,
    ),
    "gemini-3.1-pro-preview": ModelConfig(
        name="gemini-3.1-pro-preview",
        model_id="google/gemini-3.1-pro-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        temperature=0.0,
        # Budget covers thinking + structured rewrite output without truncation.
        # Older 2048 cap caused mid-JSON truncation → parse_error in ~30% of patients.
        max_tokens=8192,
        extra_body={"google": {"thinkingConfig": {"thinkingBudget": 0}}},
        input_price_per_1m=2.00,
        output_price_per_1m=12.00,
    ),
    "claude-sonnet-4-5": ModelConfig(
        name="claude-sonnet-4-5",
        model_id="claude-sonnet-4-5-20251022",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.0,
        max_tokens=2048,
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    ),
    "local": ModelConfig(
        name="local",
        model_id=os.environ.get("LOCAL_LLM_MODEL_ID", "local-model"),
        base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key_env="",   # no auth
        temperature=0.0,
        max_tokens=2048,
    ),
}

CONDITIONS: dict[str, dict] = {
    "baseline":         {"anchors": [],                                                   "params": []},
    "struc_belief":     {"anchors": ["post_pe", "post_first_labs", "post_imaging"],       "params": []},
    "struc_consult":    {"anchors": ["post_pe", "post_first_labs", "post_imaging"],       "params": []},
    "pushback_naive":   {"anchors": ["post_pe", "post_first_labs", "post_imaging"],       "params": []},
    "pushback_counter": {"anchors": ["post_pe", "post_first_labs", "post_imaging"],       "params": []},
}
