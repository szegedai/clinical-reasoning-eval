"""Target-vs-candidate dx dedup for compound diagnosis labels.

Models sometimes emit compound dxes like "Choledocholithiasis/Cholangitis" or
"Recurrent Peptic Ulcer Disease (PUD) / Gastritis". When such a label is the
injected target, plain exact-string dedup leaves the compound's components
("Choledocholithiasis" alone, "Acute Cholangitis") in the rest of the
differential, producing a redundant biased prior the model can discount.

Token-level dedup: split labels on non-alphabetic boundaries, drop short and
hedge tokens, and treat any shared token as a near-twin match.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z]{4,}")

_STOPWORDS = {
    "acute", "chronic", "recurrent", "early", "late",
    "possible", "probable", "suspected",
    "primary", "secondary", "with", "without",
    "disease", "syndrome", "disorder",
    "complication", "complications",
    "etiology", "uncomplicated", "complicated",
    "exacerbation",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def is_target_near_twin(candidate: str, target: str) -> bool:
    """True if candidate shares a meaningful token with target.

    Examples (target = "Choledocholithiasis/Cholangitis"):
      - "Choledocholithiasis"    -> True   (shares 'choledocholithiasis')
      - "Acute Cholangitis"      -> True   (shares 'cholangitis')
      - "Cholangitis NOS"        -> True
      - "Pancreatitis"           -> False
      - "Cholecystitis"          -> False

    Falls back to lowercase exact match if target has no meaningful tokens
    (rare — e.g. an all-acronym label like "MI" or "GI bleed").
    """
    t_tok = _tokens(target)
    c_tok = _tokens(candidate)
    if not t_tok:
        return (candidate or "").lower() == (target or "").lower()
    return bool(t_tok & c_tok)
