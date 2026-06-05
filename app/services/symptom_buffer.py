"""Event-driven medical session buffer.

Accumulates extracted medical entities across streaming chunks,
deduplicates, and provides a clean prompt for the final LLM call.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.services.medical_filter import FilterResult
from app.services.red_flag_engine import RedFlag, RedFlagResult


# ── Data Classes ────────────────────────────────────────────────────


@dataclass
class BufferedEntity:
    """A single buffered medical entity."""

    text: str              # normalized medical term
    original: str          # original text from transcript
    category: str          # "symptom" | "condition" | "medication"
    timestamp: float       # when it was detected (epoch seconds)
    chunk_index: int = 0   # which chunk it came from


@dataclass
class SessionBuffer:
    """Accumulates medical entities for a single consultation session.

    Continuously updated as new audio chunks are processed.
    Provides ``to_llm_prompt()`` to generate a clean, structured input
    that the LLM can turn into a final medical report.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    symptoms: list[BufferedEntity] = field(default_factory=list)
    conditions: list[BufferedEntity] = field(default_factory=list)
    medications: list[BufferedEntity] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    medical_sentences: list[str] = field(default_factory=list)
    transcript_chunks: list[str] = field(default_factory=list)

    chunks_processed: int = 0
    overall_urgency: str = "routine"

    # Internal dedup sets
    _seen_symptoms: set[str] = field(default_factory=set, repr=False)
    _seen_conditions: set[str] = field(default_factory=set, repr=False)
    _seen_medications: set[str] = field(default_factory=set, repr=False)

    # ── Adding entities ─────────────────────────────────────────────

    def add_symptom(
        self, normalized: str, original: str, timestamp: float, chunk_index: int = 0
    ) -> bool:
        """Add a symptom if not already seen.  Returns True if added."""
        key = normalized.lower()
        if key in self._seen_symptoms:
            return False
        self._seen_symptoms.add(key)
        self.symptoms.append(
            BufferedEntity(
                text=normalized,
                original=original,
                category="symptom",
                timestamp=timestamp,
                chunk_index=chunk_index,
            )
        )
        return True

    def add_condition(
        self, normalized: str, original: str, timestamp: float, chunk_index: int = 0
    ) -> bool:
        key = normalized.lower()
        if key in self._seen_conditions:
            return False
        self._seen_conditions.add(key)
        self.conditions.append(
            BufferedEntity(
                text=normalized,
                original=original,
                category="condition",
                timestamp=timestamp,
                chunk_index=chunk_index,
            )
        )
        return True

    def add_medication(
        self, normalized: str, original: str, timestamp: float, chunk_index: int = 0
    ) -> bool:
        key = normalized.lower()
        if key in self._seen_medications:
            return False
        self._seen_medications.add(key)
        self.medications.append(
            BufferedEntity(
                text=normalized,
                original=original,
                category="medication",
                timestamp=timestamp,
                chunk_index=chunk_index,
            )
        )
        return True

    def add_red_flags(self, result: RedFlagResult) -> None:
        """Merge red-flag result into the buffer."""
        for flag in result.flags:
            # Deduplicate by description
            if not any(f.description == flag.description for f in self.red_flags):
                self.red_flags.append(flag)
        # Escalate urgency
        rank = {"routine": 0, "urgent": 1, "emergency": 2}
        if rank.get(result.urgency, 0) > rank.get(self.overall_urgency, 0):
            self.overall_urgency = result.urgency

    def ingest_filter_result(
        self, fr: FilterResult, timestamp: float, chunk_index: int = 0
    ) -> None:
        """Ingest a complete ``FilterResult`` into the buffer."""
        for entity in fr.symptoms:
            self.add_symptom(entity.normalized, entity.original, timestamp, chunk_index)
        for entity in fr.conditions:
            self.add_condition(entity.normalized, entity.original, timestamp, chunk_index)
        for entity in fr.medications:
            self.add_medication(entity.normalized, entity.original, timestamp, chunk_index)

        for sentence in fr.sentences:
            if sentence.is_medical:
                self.medical_sentences.append(sentence.text)

        self.chunks_processed += 1

    def add_transcript_chunk(self, text: str) -> None:
        if text.strip():
            self.transcript_chunks.append(text.strip())

    # ── Output ──────────────────────────────────────────────────────

    @property
    def confidence_score(self) -> float:
        """Heuristic confidence based on entity density."""
        total = len(self.symptoms) + len(self.conditions) + len(self.medications)
        if self.chunks_processed == 0:
            return 0.0
        # More entities per chunk = higher confidence the extraction is useful
        density = total / self.chunks_processed
        return min(density / 3.0, 1.0)  # caps at 1.0 when ≥3 entities/chunk

    @property
    def is_empty(self) -> bool:
        # ✅ FIX: Don't block LLM just because keyword filter found nothing.
        # If we have any transcript text at all, let the LLM decide.
        # The old check (symptoms/conditions/medications) was too strict —
        # it blocked the LLM call whenever audio was slightly garbled and
        # the hardcoded filter missed keywords.
        return not self.transcript_chunks

    def to_dict(self) -> dict:
        """Serialize the buffer to a plain dict."""
        return {
            "session_id": self.session_id,
            "symptoms": [
                {"text": e.text, "original": e.original, "timestamp": e.timestamp}
                for e in self.symptoms
            ],
            "conditions_mentioned": [
                {"text": e.text, "original": e.original, "timestamp": e.timestamp}
                for e in self.conditions
            ],
            "medications_mentioned": [
                {"text": e.text, "original": e.original, "timestamp": e.timestamp}
                for e in self.medications
            ],
            "red_flags": [
                {
                    "description": f.description,
                    "urgency": f.urgency,
                    "pattern": f.pattern_matched,
                }
                for f in self.red_flags
            ],
            "overall_urgency": self.overall_urgency,
            "confidence_score": round(self.confidence_score, 3),
            "chunks_processed": self.chunks_processed,
        }

    def to_llm_prompt(self) -> str:
        """Generate a structured prompt for the final LLM call.

        Always sends the full raw transcript so the LLM can extract
        everything — including entities the keyword filter missed.
        """
        full_transcript = " ".join(self.transcript_chunks)

        return f"""/no_think
Extract a doctor-reviewable medical report from this Malayalam/Manglish
doctor-patient transcript. Extract only explicit facts. Do not diagnose or
guess. Put unclear items in uncertain_items.

FULL CONVERSATION TRANSCRIPT:
{full_transcript}

Return strict JSON with this schema:
{{
  "chief_complaints": [],
  "patient_reported_symptoms": [],
  "symptoms": [],
  "duration": [],
  "severity": [],
  "body_parts_mentioned": [],
  "past_conditions_mentioned": [],
  "conditions_mentioned": [],
  "doctor_observations": [],
  "doctor_confirmed_diagnosis": [],
  "medications_mentioned": [],
  "prescribed_medications": [
    {{
      "name": "",
      "dosage": "",
      "frequency": "",
      "duration": "",
      "timing": "",
      "route": "",
      "instructions": "",
      "evidence": ""
    }}
  ],
  "advice": [],
  "recommended_tests": [],
  "follow_up": [],
  "risk_flags": [],
  "uncertain_items": []
}}"""

        return f"""/no_think
You are a medical assistant. Read this doctor-patient conversation
and extract ALL medical information.

FULL CONVERSATION TRANSCRIPT:
{full_transcript}

Extract and return JSON with these exact keys:
- symptoms (list of strings — all symptoms the patient mentions)
- conditions_mentioned (list of strings — e.g. diabetes, hypertension)
- medications_mentioned (list of strings)
- body_parts_mentioned (list of strings)
- duration (list of strings — how long symptoms present)
- severity (list of strings)
- risk_flags (list of strings — anything urgent or dangerous)
- urgency_level (string — one of: routine / urgent / emergency)

Rules:
1. The conversation may be in Malayalam, Tamil, or mixed with English.
2. Only extract what is EXPLICITLY mentioned. Do not infer or assume.
3. Return ONLY valid JSON. No explanation. No text outside the JSON object.
"""
