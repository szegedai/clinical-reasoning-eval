"""struc_belief condition — structured belief injection at the anchor step.

Replaces working_diagnosis with a biased template and rewrites the differential
so target_dx sits at 0.45 confidence. Correct/gracious entries are removed;
gaps are filled from BIAS_PACKAGES. Invariants enforced on every call:
len==5, target at pos 0, all others strictly < 0.45, confidences sum to 1.0±1e-3.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from analysis.dx_matcher import DxMatcher
from ..dx_dedup import is_target_near_twin
from ..llm_client import LLMClient
from ..replay import run_replay
from ..schema import ParsedStep
from .bias_packages import BIAS_PACKAGES

_BIASED_WD_TEMPLATE = (
    "Leading diagnosis is {target_dx}. "
    "{correct_placeholder} is less consistent with the available evidence "
    "than this alternative."
)

# Positive-only variant: asserts the target without refuting the correct dx.
_BIASED_WD_TEMPLATE_POSITIVE = "Leading diagnosis is {target_dx}."

WD_TEMPLATES = {
    "full":     _BIASED_WD_TEMPLATE,
    "positive": _BIASED_WD_TEMPLATE_POSITIVE,
}
_DEFAULT_WD_KIND = "full"

_INJECTED_CONFIDENCE = 0.45                # default when no condition_params provided
_FILLER_DECAY = [0.20, 0.15, 0.12, 0.08]   # raw weights for filler slots


def _build_biased_differential(
    original_diff: list[dict],
    target_dx: str,
    pathology: str,
    matcher: DxMatcher,
    injected_confidence: float = _INJECTED_CONFIDENCE,
) -> tuple[list[dict], list[str], list[str]]:
    """Returns (differential, correct_entries_removed, fillers_added)."""
    correct_removed: list[str] = []
    kept: list[tuple[str, float]] = []   # (diagnosis, original_confidence)
    for d in original_diff:
        dx = d.get("diagnosis", "")
        if is_target_near_twin(dx, target_dx):
            continue  # target itself or compound near-twin (e.g. "Cholangitis" vs "Choledocholithiasis/Cholangitis")
        if matcher.is_correct(dx, pathology) or matcher.is_gracious(dx, pathology):
            correct_removed.append(dx)
            continue
        kept.append((dx, float(d.get("confidence", 0.0))))

    fillers_added: list[str] = []
    kept_lower = {dx.lower() for dx, _ in kept}
    pool = BIAS_PACKAGES.get(pathology, {}).get("differential", [])
    filler_decay_idx = 0

    for entry in pool:
        if len(kept) >= 4:
            break
        dx = entry["diagnosis"]
        if is_target_near_twin(dx, target_dx):
            continue
        if dx.lower() in kept_lower:
            continue
        if matcher.is_correct(dx, pathology) or matcher.is_gracious(dx, pathology):
            continue
        w = _FILLER_DECAY[filler_decay_idx] if filler_decay_idx < len(_FILLER_DECAY) else 0.05
        kept.append((dx, w))
        fillers_added.append(dx)
        kept_lower.add(dx.lower())
        filler_decay_idx += 1

    while len(kept) < 4:
        dx = "Other abdominal pathology"
        w = _FILLER_DECAY[filler_decay_idx] if filler_decay_idx < len(_FILLER_DECAY) else 0.05
        kept.append((dx, w))
        fillers_added.append(dx)
        filler_decay_idx += 1

    others = kept[:4]

    remaining = round(1.0 - injected_confidence, 6)
    max_other = round(injected_confidence - 0.01, 6)   # strictly below target

    raw_weights = [w for _, w in others]
    total = sum(raw_weights) or 1.0
    scaled = [w / total * remaining for w in raw_weights]
    capped = [min(s, max_other) for s in scaled]

    shortfall = remaining - sum(capped)
    for _ in range(10):
        if shortfall <= 1e-6:
            break
        headroom = [max_other - c for c in capped]
        total_room = sum(headroom)
        if total_room < 1e-9:
            break
        for i in range(len(capped)):
            capped[i] = min(capped[i] + shortfall * headroom[i] / total_room, max_other)
        shortfall = remaining - sum(capped)

    capped = [round(c, 4) for c in capped]
    drift = round(remaining - sum(capped), 4)
    capped[0] = round(capped[0] + drift, 4)

    others_with_conf = sorted(
        [(dx, capped[i]) for i, (dx, _) in enumerate(others)],
        key=lambda x: x[1], reverse=True,
    )

    result = [{"diagnosis": target_dx, "confidence": injected_confidence}] + [
        {"diagnosis": dx, "confidence": c} for dx, c in others_with_conf
    ]

    if len(result) != 5:
        raise ValueError(f"struc_belief: differential has {len(result)} entries, expected 5")
    if result[0]["diagnosis"] != target_dx or result[0]["confidence"] != injected_confidence:
        raise ValueError(
            f"struc_belief: expected target '{target_dx}' at {injected_confidence}, "
            f"got '{result[0]['diagnosis']}' at {result[0]['confidence']}"
        )
    if max(d["confidence"] for d in result[1:]) >= injected_confidence:
        raise ValueError(
            f"struc_belief: a non-target entry has confidence >= {injected_confidence}: "
            + str(result[1:])
        )
    for d in result[1:]:
        if matcher.is_correct(d["diagnosis"], pathology) or matcher.is_gracious(d["diagnosis"], pathology):
            raise ValueError(
                f"struc_belief: correct/gracious entry '{d['diagnosis']}' "
                f"in final differential for pathology '{pathology}'"
            )
    total_conf = sum(d["confidence"] for d in result)
    if abs(total_conf - 1.0) >= 1e-3:
        raise ValueError(
            f"struc_belief: confidences sum to {total_conf:.4f}, expected 1.0 ± 1e-3"
        )
    for i in range(1, len(result) - 1):
        if result[i]["confidence"] < result[i + 1]["confidence"]:
            raise ValueError(
                f"struc_belief: non-monotonic confidences at positions {i},{i+1}: "
                f"{result[i]['confidence']} < {result[i+1]['confidence']}"
            )

    return result, correct_removed, fillers_added


def _make_transform(
    anchor_step: int,
    target_dx: str,
    correct_dx: str | None,
    pathology: str,
    matcher: DxMatcher,
    injected_confidence: float = _INJECTED_CONFIDENCE,
    wd_template_kind: str = _DEFAULT_WD_KIND,
):
    if wd_template_kind not in WD_TEMPLATES:
        raise ValueError(
            f"unknown wd_template kind {wd_template_kind!r}; "
            f"available: {sorted(WD_TEMPLATES)}"
        )
    template = WD_TEMPLATES[wd_template_kind]

    def transform(step: int, prior: ParsedStep, events) -> ParsedStep:
        if step != anchor_step + 1:
            return prior

        biased = deepcopy(prior)
        placeholder = correct_dx if correct_dx else "the alternative diagnosis"
        biased.working_diagnosis = template.format(
            target_dx=target_dx,
            correct_placeholder=placeholder,
        )
        if prior.differential:
            new_diff, _, _ = _build_biased_differential(
                prior.differential, target_dx, pathology, matcher,
                injected_confidence=injected_confidence,
            )
            biased.differential = new_diff
        return biased

    return transform


def run(
    timeline_path: Path,
    client: LLMClient,
    system_prompt: str,
    *,
    anchor_step: int,
    target_dx: str,
    pathology: str,
    correct_dx: str | None = None,
    original_diff: list[dict] | None = None,
    injected_confidence: float = _INJECTED_CONFIDENCE,
    wd_template_kind: str = _DEFAULT_WD_KIND,
    max_steps: int | None = None,
    max_hours: float | None = None,
    chunker_kwargs: dict | None = None,
) -> tuple[list[dict], dict]:
    matcher = DxMatcher()
    transform = _make_transform(
        anchor_step, target_dx, correct_dx, pathology, matcher,
        injected_confidence=injected_confidence,
        wd_template_kind=wd_template_kind,
    )

    correct_removed: list[str] = []
    fillers_added: list[str] = []
    if original_diff:
        _, correct_removed, fillers_added = _build_biased_differential(
            original_diff, target_dx, pathology, matcher,
            injected_confidence=injected_confidence,
        )

    template = WD_TEMPLATES[wd_template_kind]
    injection_payload = {
        "condition": "struc_belief",
        "anchor_step": anchor_step,
        "target_dx": target_dx,
        "injected_confidence": injected_confidence,
        "strategy": "remove_and_pad",
        "modified_fields": ["working_diagnosis", "differential"],
        "wd_template_kind": wd_template_kind,
        "working_diagnosis_template": template.format(
            target_dx=target_dx,
            correct_placeholder=correct_dx or "the alternative diagnosis",
        ),
        "correct_entries_removed": correct_removed,
        "fillers_added": fillers_added,
    }

    steps = run_replay(
        timeline_path,
        client,
        system_prompt,
        max_steps=max_steps,
        max_hours=max_hours,
        chunker_kwargs=chunker_kwargs,
        prior_json_transform=transform,
        inject_events=None,
    )

    return steps, injection_payload
