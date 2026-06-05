"""Deterministic medical term lexicon for multilingual normalization.

Maps Malayalam, Manglish, and English slang/colloquial terms to
standardized medical terminology.  Pure dictionary lookup — no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Term Mappings ───────────────────────────────────────────────────
# Keys are LOWERCASE.  Values are canonical medical terms.

SYMPTOM_LEXICON: dict[str, str] = {
    # English common
    "fever": "fever",
    "high fever": "high-grade fever",
    "mild fever": "low-grade fever",
    "headache": "headache",
    "head pain": "headache",
    "stomach pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "tummy pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "chest pain": "chest pain",
    "back pain": "back pain",
    "body pain": "generalized body ache",
    "joint pain": "arthralgia",
    "knee pain": "knee pain",
    "throat pain": "sore throat",
    "sore throat": "sore throat",
    "cough": "cough",
    "dry cough": "dry cough",
    "wet cough": "productive cough",
    "cold": "common cold",
    "running nose": "rhinorrhea",
    "runny nose": "rhinorrhea",
    "blocked nose": "nasal congestion",
    "sneezing": "sneezing",
    "breathlessness": "dyspnea",
    "breathing difficulty": "dyspnea",
    "shortness of breath": "dyspnea",
    "difficulty breathing": "dyspnea",
    "wheezing": "wheezing",
    "vomiting": "vomiting",
    "nausea": "nausea",
    "diarrhea": "diarrhea",
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "constipation": "constipation",
    "dizziness": "dizziness",
    "giddiness": "dizziness",
    "fainting": "syncope",
    "unconscious": "loss of consciousness",
    "unconsciousness": "loss of consciousness",
    "tiredness": "fatigue",
    "fatigue": "fatigue",
    "weakness": "generalized weakness",
    "weight loss": "weight loss",
    "weight gain": "weight gain",
    "swelling": "edema",
    "itching": "pruritus",
    "rash": "skin rash",
    "skin rash": "skin rash",
    "bleeding": "bleeding",
    "blood in stool": "hematochezia",
    "blood in urine": "hematuria",
    "burning urination": "dysuria",
    "frequent urination": "polyuria",
    "excessive thirst": "polydipsia",
    "excessive hunger": "polyphagia",
    "blurred vision": "blurred vision",
    "eye pain": "ocular pain",
    "ear pain": "otalgia",
    "hearing loss": "hearing loss",
    "numbness": "numbness",
    "tingling": "paresthesia",
    "palpitations": "palpitations",
    "chest tightness": "chest tightness",
    "anxiety": "anxiety",
    "depression": "depression",
    "insomnia": "insomnia",
    "sleeplessness": "insomnia",
    "acidity": "gastric hyperacidity",
    "gas": "flatulence",
    "bloating": "abdominal bloating",
    "seizure": "seizure",
    "fits": "seizure",
    "convulsion": "seizure",
    "paralysis": "paralysis",

    # ── Malayalam / Manglish colloquial ──────────────────────────────
    "pani": "fever",
    "pani adikkunu": "fever",
    "thala vedana": "headache",
    "thalavedana": "headache",
    "vayaru vedana": "abdominal pain",
    "vayaruvedana": "abdominal pain",
    "neeru vedana": "chest pain",
    "maaridanam vedana": "chest pain",
    "chuma": "cough",
    "chumal": "cough",
    "thumpikkunnu": "sneezing",
    "mooku adayunnu": "nasal congestion",
    "mooku ozhukunnu": "rhinorrhea",
    "ookkal": "vomiting",
    "manasspikkal": "nausea",
    "vayaru ilakunnu": "diarrhea",
    "thalachuttam": "dizziness",
    "kshenam": "fatigue",
    "kshinam": "fatigue",
    "neer veeppam": "edema",
    "chorichil": "pruritus",
    "rakthasravam": "bleeding",
    "swasam muttunnu": "dyspnea",
    "swasam kittunnilla": "dyspnea",
    "nidraillayma": "insomnia",
    "urakkamizhma": "insomnia",
    "thalavedhana": "headache",
    "meivedana": "generalized body ache",
    "nerv": "anxiety",

    # ── Malayalam Unicode ─────────────────────────────────────────────
    "പനി": "fever",
    "തലവേദന": "headache",
    "ചുമ": "cough",
    "ഛർദ്ദി": "vomiting",
    "നെഞ്ചുവേദന": "chest pain",
    "ശ്വാസതടസ്സം": "dyspnea",
    "തലകറക്കം": "dizziness",
    "ക്ഷീണം": "fatigue",
}

CONDITION_LEXICON: dict[str, str] = {
    # English slang → standard
    "sugar": "diabetes mellitus",
    "sugar problem": "diabetes mellitus",
    "high sugar": "hyperglycemia",
    "low sugar": "hypoglycemia",
    "feeling low sugar": "hypoglycemia",
    "bp": "hypertension",
    "high bp": "hypertension",
    "low bp": "hypotension",
    "pressure": "hypertension",
    "pressure high": "hypertension",
    "pressure low": "hypotension",
    "blood pressure": "hypertension",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "heart attack": "myocardial infarction",
    "heart problem": "cardiovascular disease",
    "kidney problem": "renal disease",
    "kidney stone": "nephrolithiasis",
    "liver problem": "hepatic disease",
    "thyroid": "thyroid disorder",
    "thyroid problem": "thyroid disorder",
    "asthma": "bronchial asthma",
    "tb": "tuberculosis",
    "cancer": "malignancy",
    "stroke": "cerebrovascular accident",
    "dengue": "dengue fever",
    "malaria": "malaria",
    "typhoid": "typhoid fever",
    "jaundice": "jaundice",
    "pneumonia": "pneumonia",
    "covid": "COVID-19",
    "corona": "COVID-19",

    # Malayalam / Manglish
    "prameham": "diabetes mellitus",
    "sugar rogam": "diabetes mellitus",
    "rakthasamardham": "hypertension",
    "hridayaghatam": "myocardial infarction",
    "shwasakosham": "bronchial asthma",
    "maduppu": "diabetes mellitus",

    # ── Malayalam Unicode ─────────────────────────────────────────────
    "മധുമേഹം": "diabetes mellitus",
    "രക്തസമ്മർദ്ദം": "hypertension",
}

MEDICATION_LEXICON: dict[str, str] = {
    # Common abbreviations & slang
    "paracetamol": "paracetamol",
    "dolo": "paracetamol (Dolo-650)",
    "dolo 650": "paracetamol (Dolo-650)",
    "crocin": "paracetamol (Crocin)",
    "combiflam": "ibuprofen + paracetamol (Combiflam)",
    "azithromycin": "azithromycin",
    "amoxicillin": "amoxicillin",
    "metformin": "metformin",
    "insulin": "insulin",
    "bp tablet": "antihypertensive",
    "bp medicine": "antihypertensive",
    "sugar tablet": "oral hypoglycemic",
    "sugar medicine": "oral hypoglycemic",
    "cough syrup": "antitussive syrup",
    "antacid": "antacid",
    "pantoprazole": "pantoprazole",
    "omeprazole": "omeprazole",
    "cetrizine": "cetirizine",
    "cetirizine": "cetirizine",
    "montair": "montelukast (Montair)",
    "aspirin": "aspirin",
    "ecosprin": "aspirin (Ecosprin)",
    "atorvastatin": "atorvastatin",
    "amlodipine": "amlodipine",
    "losartan": "losartan",
    "enalapril": "enalapril",
    "antibiotic": "antibiotic",
    "pain killer": "analgesic",
    "painkiller": "analgesic",
    "steroid": "corticosteroid",
}

# ── Noise Keywords ──────────────────────────────────────────────────
# Phrases that indicate non-medical conversational filler.

NOISE_PHRASES: set[str] = {
    "hello",
    "hi",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how do you do",
    "fine thank you",
    "please sit",
    "please sit down",
    "come in",
    "thank you",
    "thanks",
    "thank you doctor",
    "ok doctor",
    "okay doctor",
    "yes doctor",
    "no doctor",
    "bye",
    "goodbye",
    "see you",
    "take care",
    "have a nice day",
    "namaskaram",
    "sugamano",
    "sugam aano",
    "enthaanu vishesham",
    "enthanu vishesham",
    "sheriyaakum",
    "seri aakum",
    "pinne kaanam",
    "nanni",
    "nanni doctor",
    "sherikum nanni",
}


# ── Lookup Functions ────────────────────────────────────────────────


@dataclass
class LexiconMatch:
    """A single matched term with its source and normalized form."""

    original: str
    normalized: str
    category: str  # "symptom" | "condition" | "medication"
    start_pos: int = 0
    end_pos: int = 0


def normalize_term(text: str) -> str | None:
    """Look up a single term across all lexicons.

    Returns the normalized medical term, or ``None`` if not found.
    """
    key = text.strip().lower()
    return (
        SYMPTOM_LEXICON.get(key)
        or CONDITION_LEXICON.get(key)
        or MEDICATION_LEXICON.get(key)
    )


def _scan_lexicon(
    text: str,
    lexicon: dict[str, str],
    category: str,
) -> list[LexiconMatch]:
    """Scan *text* for all entries in *lexicon* (longest-match-first)."""
    lower = text.lower()
    matches: list[LexiconMatch] = []
    # Sort by key length descending so longer phrases match first
    for key in sorted(lexicon, key=len, reverse=True):
        # Word-boundary match to avoid partial hits (supports English & Malayalam Unicode)
        pattern = rf"(?<![\w\u0D00-\u0D7F]){re.escape(key)}(?![\w\u0D00-\u0D7F])"
        for m in re.finditer(pattern, lower):
            matches.append(
                LexiconMatch(
                    original=text[m.start() : m.end()],
                    normalized=lexicon[key],
                    category=category,
                    start_pos=m.start(),
                    end_pos=m.end(),
                )
            )
    return matches


def scan_all(text: str) -> list[LexiconMatch]:
    """Scan *text* against symptom, condition, and medication lexicons.

    Returns a deduplicated list of ``LexiconMatch`` objects.
    """
    all_matches: list[LexiconMatch] = []
    all_matches.extend(_scan_lexicon(text, SYMPTOM_LEXICON, "symptom"))
    all_matches.extend(_scan_lexicon(text, CONDITION_LEXICON, "condition"))
    all_matches.extend(_scan_lexicon(text, MEDICATION_LEXICON, "medication"))

    # Deduplicate by (normalized, category)
    seen: set[tuple[str, str]] = set()
    unique: list[LexiconMatch] = []
    for match in all_matches:
        key = (match.normalized, match.category)
        if key not in seen:
            seen.add(key)
            unique.append(match)
    return unique


def is_noise(sentence: str) -> bool:
    """Return ``True`` if *sentence* is conversational noise."""
    cleaned = sentence.strip().lower().rstrip(".!?,;:")
    if cleaned in NOISE_PHRASES:
        return True
    # Also check if the sentence is very short and matches a noise prefix
    for noise in NOISE_PHRASES:
        if cleaned.startswith(noise) and len(cleaned) < len(noise) + 15:
            return True
    return False
