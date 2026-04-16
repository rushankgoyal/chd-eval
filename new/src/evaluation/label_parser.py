"""
CHD Label Parser
================
Extracts a CHD class label from raw model output text using a 5-layer fallback
strategy designed to handle diverse output formats (bare labels, CoT reasoning
blocks, JSON, long-form medical descriptions).

The parser is intentionally conservative: it returns ``None`` rather than
guessing when no label can be reliably identified.
"""

from __future__ import annotations

import json
import re
from typing import Optional

LABELS: list[str] = ["ASD", "VSD", "PDA", "Normal"]

# Canonical long-form names and common synonyms that models may produce.
LABEL_ALIASES: dict[str, list[str]] = {
    "ASD": [
        "asd",
        "atrial septal defect",
        "atrial septal",
        "ostium secundum",
        "secundum asd",
        "right heart enlargement",
    ],
    "VSD": [
        "vsd",
        "ventricular septal defect",
        "ventricular septal",
        "interventricular communication",
        "interventricular defect",
    ],
    "PDA": [
        "pda",
        "patent ductus arteriosus",
        "ductus arteriosus",
        "ductal",
        "patent duct",
    ],
    "Normal": [
        "normal",
        "no chd",
        "no congenital",
        "no defect",
        "no abnormality",
        "healthy",
        "unremarkable",
        "within normal limits",
        "no significant abnormality",
    ],
}


def parse_predicted_label(text: str) -> Optional[str]:
    """Extract a CHD class label from raw model output text.

    Applies the following strategies in order, returning on first success:

    1. The entire response (stripped) exactly matches a label (case-insensitive).
    2. ``DIAGNOSIS: <label>`` pattern (CoT prompt format).
    3. JSON object containing a ``"diagnosis"`` / ``"label"`` / similar key.
    4. Long-form synonym or alias match anywhere in the text.
    5. Last bare occurrence of ASD / VSD / PDA / Normal in the text.

    Returns ``None`` if no label can be extracted.
    """
    if not text:
        return None

    stripped = text.strip()

    # 1. Exact match against the entire stripped response
    for label in LABELS:
        if stripped.lower() == label.lower():
            return label

    # 2. "DIAGNOSIS: <label>" pattern (CoT output format)
    m = re.search(r"DIAGNOSIS\s*:\s*(\w[\w\s]*)", stripped, re.IGNORECASE)
    if m:
        matched = _match_alias(m.group(1).strip())
        if matched:
            return matched

    # 3. JSON structure (models that produce structured output)
    json_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            for key in ("diagnosis", "label", "classification", "prediction", "class", "result"):
                if key in obj:
                    matched = _match_alias(str(obj[key]))
                    if matched:
                        return matched
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Synonym / long-form match anywhere in the text
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", stripped, re.IGNORECASE):
                return label

    # 5. Last bare label occurrence (covers most CoT free-form endings)
    matches = re.findall(r"\b(ASD|VSD|PDA|Normal|NORMAL)\b", stripped, re.IGNORECASE)
    if matches:
        candidate = matches[-1]
        if candidate.upper() in ("ASD", "VSD", "PDA"):
            return candidate.upper()
        if candidate.lower() == "normal":
            return "Normal"

    return None


def _match_alias(text: str) -> Optional[str]:
    """Return the canonical LABEL that ``text`` aliases to, or None."""
    text_lower = text.strip().lower()
    for label, aliases in LABEL_ALIASES.items():
        if text_lower == label.lower() or text_lower in aliases:
            return label
    return None
