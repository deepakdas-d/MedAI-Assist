"""Rule-based red flag / emergency detection engine.

Pure pattern matching — runs in <5 ms.  No LLM.
Returns urgency levels and specific flag descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Red Flag Definitions ───────────────────────────────────────────
# Each entry: (pattern_list, flag_label, urgency)
# Urgency: "emergency" | "urgent" | "routine"

_RED_FLAG_RULES: list[tuple[list[str], str, str]] = [
    # ── EMERGENCY (immediate action required) ───────────────────────
    (
        ["chest pain", "chest tightness", "neeru vedana", "maaridanam vedana"],
        "Possible cardiac event — chest pain reported",
        "emergency",
    ),
    (
        ["unconscious", "unconsciousness", "loss of consciousness", "fainted", "collapsed"],
        "Loss of consciousness — immediate assessment needed",
        "emergency",
    ),
    (
        [
            "difficulty breathing",
            "cannot breathe",
            "can't breathe",
            "breathless",
            "severe breathlessness",
            "swasam muttunnu",
            "swasam kittunnilla",
        ],
        "Severe respiratory distress",
        "emergency",
    ),
    (
        ["seizure", "seizures", "fits", "convulsion", "convulsions"],
        "Seizure activity reported",
        "emergency",
    ),
    (
        ["severe bleeding", "heavy bleeding", "uncontrolled bleeding", "bleeding profusely"],
        "Severe hemorrhage",
        "emergency",
    ),
    (
        ["stroke", "paralysis", "one side numb", "face drooping", "slurred speech"],
        "Possible cerebrovascular event (stroke signs)",
        "emergency",
    ),
    (
        ["anaphylaxis", "severe allergic", "throat swelling", "tongue swelling"],
        "Possible anaphylactic reaction",
        "emergency",
    ),
    (
        ["suicide", "suicidal", "want to die", "kill myself", "ending life"],
        "Suicidal ideation — psychiatric emergency",
        "emergency",
    ),

    # ── URGENT (needs prompt attention) ─────────────────────────────
    (
        ["high fever", "very high fever", "104", "105", "106"],
        "High-grade fever — monitor closely",
        "urgent",
    ),
    (
        ["blood in stool", "blood in urine", "hematuria", "hematochezia", "blood vomiting"],
        "Bleeding from orifice — urgent workup needed",
        "urgent",
    ),
    (
        ["severe headache", "worst headache", "thunderclap headache", "sudden headache"],
        "Severe/sudden headache — rule out intracranial pathology",
        "urgent",
    ),
    (
        ["severe abdominal pain", "acute abdomen", "sharp stomach pain"],
        "Acute abdominal pain — urgent evaluation",
        "urgent",
    ),
    (
        ["diabetic ketoacidosis", "dka", "very high sugar", "sugar very high"],
        "Possible diabetic emergency",
        "urgent",
    ),
    (
        ["dehydration", "severe dehydration", "not drinking water", "dry mouth sunken eyes"],
        "Dehydration signs",
        "urgent",
    ),
    (
        ["palpitations", "irregular heartbeat", "heart racing", "fast heartbeat"],
        "Cardiac arrhythmia signs",
        "urgent",
    ),
]


# ── Data Classes ────────────────────────────────────────────────────


@dataclass
class RedFlag:
    """A single triggered red flag."""

    pattern_matched: str
    description: str
    urgency: str  # "emergency" | "urgent"


@dataclass
class RedFlagResult:
    """Result of running the red flag engine on text."""

    flags: list[RedFlag] = field(default_factory=list)
    urgency: str = "routine"  # overall urgency = max of all flags

    @property
    def has_flags(self) -> bool:
        return len(self.flags) > 0

    @property
    def is_emergency(self) -> bool:
        return self.urgency == "emergency"


# ── Engine ──────────────────────────────────────────────────────────

_URGENCY_RANK = {"routine": 0, "urgent": 1, "emergency": 2}


def check_red_flags(text: str) -> RedFlagResult:
    """Scan *text* for red flag patterns and return the result.

    Parameters
    ----------
    text:
        Any transcript text to scan (can be partial chunk or full).

    Returns
    -------
    RedFlagResult
        All triggered flags and the overall urgency level.
    """
    lower = text.lower()
    result = RedFlagResult()
    max_urgency = "routine"

    for patterns, description, urgency in _RED_FLAG_RULES:
        for pattern in patterns:
            if re.search(rf"\b{re.escape(pattern)}\b", lower):
                result.flags.append(
                    RedFlag(
                        pattern_matched=pattern,
                        description=description,
                        urgency=urgency,
                    )
                )
                if _URGENCY_RANK[urgency] > _URGENCY_RANK[max_urgency]:
                    max_urgency = urgency
                break  # one match per rule is enough

    result.urgency = max_urgency
    return result
