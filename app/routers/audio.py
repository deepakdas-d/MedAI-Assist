"""Audio processing endpoints — transcribe and analyse medical conversations."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, UploadFile

from app.schemas import (
    AnalysisResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    DiagnosisCandidate,
    DiagnosisResult,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MedicalExtraction,
    MedicineItem,
    SegmentOut,
    TranscriptionResponse,
)
from app.services.extraction import diagnose_from_symptoms, extract_medical_info
from app.services.audio_preprocessing import cleanup_prepared_audio, prepare_audio_for_whisper
from app.whisper_client import whisper_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["audio"])


# ── Helpers ─────────────────────────────────────────────────────────


def _save_upload(file: UploadFile) -> str:
    """Persist an uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file.file.read())
    finally:
        tmp.close()
    return tmp.name


def _parse_extraction(data: dict) -> MedicalExtraction:
    """Safely parse Qwen3 JSON output into a MedicalExtraction model."""
    prescribed = []
    for item in data.get("prescribed_medications", []):
        if isinstance(item, dict):
            prescribed.append(MedicineItem(**item))

    return MedicalExtraction(
        chief_complaints=data.get("chief_complaints", []),
        patient_reported_symptoms=data.get(
            "patient_reported_symptoms", data.get("symptoms", [])
        ),
        symptoms=data.get("symptoms", []),
        past_conditions_mentioned=data.get(
            "past_conditions_mentioned", data.get("conditions_mentioned", [])
        ),
        conditions_mentioned=data.get("conditions_mentioned", []),
        medications_mentioned=data.get("medications_mentioned", []),
        prescribed_medications=prescribed,
        body_parts_mentioned=data.get("body_parts_mentioned", []),
        duration=data.get("duration", []),
        severity=data.get("severity", []),
        doctor_observations=data.get("doctor_observations", []),
        doctor_confirmed_diagnosis=data.get("doctor_confirmed_diagnosis", []),
        advice=data.get("advice", []),
        recommended_tests=data.get("recommended_tests", []),
        follow_up=data.get("follow_up", []),
        risk_flags=data.get("risk_flags", []),
        uncertain_items=data.get("uncertain_items", []),
    )


def _parse_diagnosis(data: dict) -> DiagnosisResult:
    """Safely parse Qwen3 JSON output into a DiagnosisResult model."""
    candidates_raw = data.get("candidates", [])
    candidates = []
    for c in candidates_raw:
        if isinstance(c, dict):
            candidates.append(DiagnosisCandidate(**c))

    return DiagnosisResult(
        candidates=candidates,
        recommended_tests=data.get("recommended_tests", []),
        urgency_level=data.get("urgency_level", "routine"),
        reasoning_summary=data.get("reasoning_summary", ""),
    )


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: Annotated[
        str | None,
        Query(
            description=(
                'ISO-639-1 code to force (e.g. "ml" for Malayalam). '
                'Default is forced Malayalam for prototype stability. Use "auto" to test auto-detect.'
            )
        ),
    ] = None,
) -> TranscriptionResponse:
    """Upload an audio file and get a transcript.

    - **Prototype default**: forced Malayalam (`ml`) to avoid Tamil/gibberish demo output.
    - **Auto-detect test**: pass `language=auto` only when comparing real mixed-language audio.
    """
    tmp_path = _save_upload(file)
    prepared = prepare_audio_for_whisper(tmp_path)
    try:
        result = await whisper_client.transcribe(prepared.path, language=language)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_prepared_audio(prepared)
        os.unlink(tmp_path)

    # Remote API returns 'segments' as list of dicts, which matches SegmentOut
    # Local fallback also returns a similar structure
    return TranscriptionResponse(
        transcript=result.get("transcript", result.get("text", "")),
        detected_language=result.get("language", ""),
        language_probability=result.get("language_probability", 0.0),
        audio_quality=prepared.quality,
        segments=[
            SegmentOut(start=s["start"], end=s["end"], text=s["text"])
            for s in result.get("segments", [])
        ],
    )


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile,
    language: Annotated[
        str | None,
        Query(
            description=(
                'ISO-639-1 code to force (e.g. "ml" for Malayalam). '
                'Default is forced Malayalam for prototype stability. Use "auto" to test auto-detect.'
            )
        ),
    ] = None,
) -> AnalysisResponse:
    """Upload audio → transcribe → extract structured medical info via Qwen3.

    Uses **no-think mode** for fast, accurate structured extraction.
    Requires Ollama running locally with the configured model.
    """
    tmp_path = _save_upload(file)
    prepared = prepare_audio_for_whisper(tmp_path)
    try:
        result = await whisper_client.transcribe(prepared.path, language=language)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_prepared_audio(prepared)
        os.unlink(tmp_path)

    transcript = result.get("transcript", result.get("text", ""))
    logger.info("Transcriptions completed. Starting medical extraction via Qwen3...")
    try:
        raw_extraction = await extract_medical_info(transcript)
        extraction = _parse_extraction(raw_extraction)
    except Exception as exc:
        logger.exception("Medical extraction failed (is Ollama running?)")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama/Qwen3 extraction failed: {exc}",
        ) from exc

    logger.info("Total processing for /analyze finished successfully.")
    return AnalysisResponse(
        transcript=transcript,
        detected_language=result.get("language", ""),
        audio_quality=prepared.quality,
        extraction=extraction,
    )


@router.post("/extract")
async def extract_from_text(body: ExtractRequest) -> ExtractResponse:
    """Extract structured medical info from a transcript (text only, no audio).

    Uses **no-think mode** (/no_think) — fast structured parsing without
    chain-of-thought reasoning. Best for extracting symptoms, medicines,
    diagnosis, and advice from an existing transcript.
    """
    try:
        raw = await extract_medical_info(body.transcript)
        extraction = _parse_extraction(raw)
    except Exception as exc:
        logger.exception("Medical extraction failed")
        raise HTTPException(
            status_code=502,
            detail=f"Extraction failed: {exc}",
        ) from exc

    from app.config import OLLAMA_MODEL

    return ExtractResponse(extraction=extraction, model=OLLAMA_MODEL, mode="no_think")


@router.post("/diagnose")
async def diagnose_symptoms(body: DiagnoseRequest) -> DiagnoseResponse:
    """Diagnose from patient symptoms using deep reasoning.

    Uses **think mode** (/think) — Qwen3 performs multi-step chain-of-thought
    reasoning to generate differential diagnoses with confidence levels.
    Slower but more accurate for complex/vague symptom analysis.
    """
    try:
        raw = await diagnose_from_symptoms(body.symptoms, body.patient_context)
        diagnosis = _parse_diagnosis(raw)
    except Exception as exc:
        logger.exception("Diagnosis reasoning failed")
        raise HTTPException(
            status_code=502,
            detail=f"Diagnosis failed: {exc}",
        ) from exc

    from app.config import OLLAMA_MODEL

    return DiagnoseResponse(diagnosis=diagnosis, model=OLLAMA_MODEL, mode="think")


