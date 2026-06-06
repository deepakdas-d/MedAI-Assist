"""Safety cleanup for extraction results from low-confidence transcripts."""

from __future__ import annotations

import re
from copy import deepcopy


REPORT_FIELDS = [
    "chief_complaints",
    "patient_reported_symptoms",
    "symptoms",
    "duration",
    "severity",
    "body_parts_mentioned",
    "past_conditions_mentioned",
    "conditions_mentioned",
    "doctor_observations",
    "doctor_confirmed_diagnosis",
    "medications_mentioned",
    "prescribed_medications",
    "advice",
    "recommended_tests",
    "follow_up",
    "risk_flags",
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0d00-\u0d7f\u0b80-\u0bff\u0c80-\u0cff]+", text.lower())


def transcript_has_asr_artifacts(text: str) -> bool:
    """Detect severe ASR artifacts without rejecting valid mixed speech."""

    tokens = _tokens(text)
    if not tokens:
        return True

    repeated_ratio = 1.0 - (len(set(tokens)) / len(tokens))
    punctuation_runs = len(re.findall(r"(?:[,।.!?]\s*){4,}", text))
    malayalam_chars = len(re.findall(r"[\u0d00-\u0d7f]", text))
    tamil_kannada_chars = len(re.findall(r"[\u0b80-\u0bff\u0c80-\u0cff]", text))
    replacement_chars = text.count("\ufffd")

    return (
        replacement_chars > 0
        or punctuation_runs > 0
        or (len(tokens) >= 8 and repeated_ratio >= 0.55)
        or (tamil_kannada_chars >= 20 and malayalam_chars == 0)
    )


def apply_transcript_safety(transcript: str, extraction: dict) -> dict:
    """Prevent hallucinated report fields when transcription obviously failed."""

    if not transcript_has_asr_artifacts(transcript):
        return extraction

    safe = deepcopy(extraction) if isinstance(extraction, dict) else {}
    for field in REPORT_FIELDS:
        safe[field] = []

    uncertain = safe.get("uncertain_items", [])
    if not isinstance(uncertain, list):
        uncertain = [str(uncertain)]

    note = (
        "Transcript appears low-confidence or corrupted by ASR artifacts; "
        "medical fields were left empty for doctor review."
    )
    if note not in uncertain:
        uncertain.insert(0, note)

    fragment = transcript.strip()
    if fragment:
        fragment_note = f"Raw uncertain transcript: {fragment[:300]}"
        if fragment_note not in uncertain:
            uncertain.append(fragment_note)

    safe["uncertain_items"] = uncertain
    return safe
