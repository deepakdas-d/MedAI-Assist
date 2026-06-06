"""Medical information extraction via Ollama / Qwen3.

The main prototype path uses /no_think for fast extraction-only report
generation. Diagnosis reasoning is kept as a separate optional endpoint.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import OLLAMA_CHAT_URL, OLLAMA_HTTPX_TIMEOUT, OLLAMA_MODEL

logger = logging.getLogger(__name__)


EXTRACT_PROMPT = """/no_think
You extract a doctor-reviewable medical report from a Malayalam/Manglish
doctor-patient transcript.

Task rules:
- Extract only explicit medical facts from the transcript.
- Do not diagnose, suggest diseases, or infer missing symptoms.
- Ignore greetings, jokes, casual talk, and unrelated conversation.
- Keep patient complaints separate from doctor advice/treatment.
- If a value is unclear, put it in uncertain_items instead of guessing.
- Normalize common terms: sugar = diabetes context, BP/pressure = blood pressure context.
- Return strict JSON only.

Transcript:
{transcript}

JSON schema:
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


DIAGNOSE_PROMPT = """/think
You are a physician-assist reasoning tool. This output is not a final diagnosis.

Patient Symptoms:
{symptoms}

Patient Context:
{context}

Return JSON with exactly these keys:
- "candidates": list of objects with "condition", "confidence" (high/medium/low), "reasoning"
- "recommended_tests": list of test names to confirm
- "urgency_level": one of "emergency", "urgent", "routine"
- "reasoning_summary": short summary

Return only valid JSON."""


async def _call_ollama(prompt: str, json_mode: bool = True) -> str:
    """Send a chat message to Qwen3 via Ollama and return the response text."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0},
    }
    if json_mode:
        payload["format"] = "json"

    from app.config import OLLAMA_API_URL

    target = "Remote Colab GPU" if OLLAMA_API_URL else "Local CPU"
    logger.info(
        "Calling Ollama (%s) on %s with %d-char prompt",
        OLLAMA_MODEL,
        target,
        len(prompt),
    )

    async with httpx.AsyncClient(timeout=OLLAMA_HTTPX_TIMEOUT) as client:
        response = await client.post(OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()

    result: str = response.json()["message"]["content"]
    logger.info("Qwen3 responded (%d chars)", len(result))
    return result


def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output, stripping wrapper text."""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    return json.loads(text)


async def extract_medical_info(transcript: str) -> dict:
    """Extract structured report fields from a transcript using /no_think."""

    prompt = EXTRACT_PROMPT.format(transcript=transcript)
    raw = await _call_ollama(prompt, json_mode=True)
    return _extract_json(raw)


async def diagnose_from_symptoms(
    symptoms: list[str],
    patient_context: str = "",
) -> dict:
    """Generate optional physician-assist differential reasoning."""

    symptoms_text = "\n".join(f"- {s}" for s in symptoms)
    context = patient_context or "No additional context provided."
    prompt = DIAGNOSE_PROMPT.format(symptoms=symptoms_text, context=context)
    raw = await _call_ollama(prompt, json_mode=True)
    return _extract_json(raw)


async def structure_from_buffer(buffer_prompt: str) -> dict:
    """Generate a final structured report from a streaming buffer prompt."""

    raw = await _call_ollama(buffer_prompt, json_mode=True)
    return _extract_json(raw)
