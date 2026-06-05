"""Lightweight streaming medical entity filter.

Runs on every partial transcript chunk (<10 ms).
Extracts medical entities using the lexicon and classifies sentences
as ``medical`` or ``noise`` via keyword-density scoring — **no LLM**.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.medical_lexicon import LexiconMatch, is_noise, scan_all

# ── Thresholds ──────────────────────────────────────────────────────

_MIN_MEDICAL_KEYWORD_DENSITY: float = 0.05  # 5 % of words must be medical


# ── Data Classes ────────────────────────────────────────────────────


@dataclass
class FilteredSentence:
    """Classification result for a single sentence."""

    text: str
    is_medical: bool
    entities: list[LexiconMatch] = field(default_factory=list)


@dataclass
class FilterResult:
    """Output of filtering a transcript chunk."""

    sentences: list[FilteredSentence] = field(default_factory=list)
    symptoms: list[LexiconMatch] = field(default_factory=list)
    conditions: list[LexiconMatch] = field(default_factory=list)
    medications: list[LexiconMatch] = field(default_factory=list)
    noise_count: int = 0
    medical_count: int = 0

    @property
    def has_medical_content(self) -> bool:
        return self.medical_count > 0


# ── Core Filter ─────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common delimiters."""
    parts = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in parts if s.strip()]


def _classify_sentence(sentence: str) -> FilteredSentence:
    """Classify a single sentence as medical or noise."""
    # Fast noise check first
    if is_noise(sentence):
        return FilteredSentence(text=sentence, is_medical=False)

    # Scan for medical entities
    entities = scan_all(sentence)

    if entities:
        return FilteredSentence(text=sentence, is_medical=True, entities=entities)

    # Keyword density check for sentences without exact lexicon matches
    words = sentence.lower().split()
    if not words:
        return FilteredSentence(text=sentence, is_medical=False)

    # Medical context words (not in lexicon but indicate medical content)
    medical_context_words = {
        "doctor",
        "hospital",
        "medicine",
        "treatment",
        "test",
        "report",
        "scan",
        "xray",
        "x-ray",
        "blood test",
        "urine test",
        "tablet",
        "capsule",
        "injection",
        "surgery",
        "operation",
        "admission",
        "discharge",
        "prescription",
        "dose",
        "dosage",
        "side effect",
        "allergy",
        "allergic",
        "checkup",
        "check-up",
        "diagnosis",
        "patient",
        "symptom",
        "pain",
        "swelling",
        "infection",
        "inflammation",
        "mg",
        "ml",
        "twice",
        "thrice",
        "daily",
        "morning",
        "evening",
        "night",
        "before food",
        "after food",
        "empty stomach",
    }

    context_count = sum(1 for w in words if w in medical_context_words)
    density = context_count / len(words)

    is_medical = density >= _MIN_MEDICAL_KEYWORD_DENSITY
    return FilteredSentence(text=sentence, is_medical=is_medical)


def filter_chunk(text: str) -> FilterResult:
    """Filter a transcript chunk and extract medical entities.

    Parameters
    ----------
    text:
        Raw partial transcript text (from one Whisper chunk).

    Returns
    -------
    FilterResult
        Classified sentences, extracted entities, and counts.
    """
    result = FilterResult()
    sentences = _split_sentences(text)

    for sentence in sentences:
        classified = _classify_sentence(sentence)
        result.sentences.append(classified)

        if classified.is_medical:
            result.medical_count += 1
            for entity in classified.entities:
                if entity.category == "symptom":
                    result.symptoms.append(entity)
                elif entity.category == "condition":
                    result.conditions.append(entity)
                elif entity.category == "medication":
                    result.medications.append(entity)
        else:
            result.noise_count += 1

    return result
