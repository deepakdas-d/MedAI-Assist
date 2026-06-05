"""Pydantic schemas for request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Shared Medical Models ───────────────────────────────────────────


class MedicineItem(BaseModel):
    """A single prescribed medicine."""

    name: str = Field(description="Medicine name")
    dosage: str = Field(default="", description="Dosage amount (e.g. 500mg)")
    frequency: str = Field(default="", description="How often (e.g. twice daily)")
    duration: str = Field(default="", description="Duration (e.g. 5 days)")
    timing: str = Field(default="", description="When to take it (e.g. after food)")
    route: str = Field(default="", description="Route such as oral, topical, injection")
    instructions: str = Field(default="", description="Extra doctor instructions")
    evidence: str = Field(default="", description="Transcript phrase supporting this item")


class MedicalExtraction(BaseModel):
    """Structured medical info extracted from a transcript."""

    chief_complaints: list[str] = Field(default_factory=list)
    patient_reported_symptoms: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    past_conditions_mentioned: list[str] = Field(default_factory=list)
    conditions_mentioned: list[str] = Field(default_factory=list)
    medications_mentioned: list[str] = Field(default_factory=list)
    prescribed_medications: list[MedicineItem] = Field(default_factory=list)
    body_parts_mentioned: list[str] = Field(default_factory=list)
    duration: list[str] = Field(default_factory=list)
    severity: list[str] = Field(default_factory=list)
    doctor_observations: list[str] = Field(default_factory=list)
    doctor_confirmed_diagnosis: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)


class DiagnosisCandidate(BaseModel):
    """A possible diagnosis with confidence and reasoning."""

    condition: str = Field(description="Name of the possible condition")
    confidence: str = Field(description="Confidence level: high / medium / low")
    reasoning: str = Field(description="Why this condition is suspected")


class DiagnosisResult(BaseModel):
    """Full diagnostic reasoning output."""

    candidates: list[DiagnosisCandidate] = Field(
        default_factory=list, description="Possible diagnoses ranked by confidence"
    )
    recommended_tests: list[str] = Field(
        default_factory=list, description="Tests to confirm diagnosis"
    )
    urgency_level: str = Field(
        default="routine", description="Urgency: emergency / urgent / routine"
    )
    reasoning_summary: str = Field(
        default="", description="Summary of the diagnostic reasoning chain"
    )


# ── Request Models ──────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    """Input for text-only medical extraction."""

    transcript: str = Field(description="Medical conversation transcript text")


class DiagnoseRequest(BaseModel):
    """Input for symptom-based diagnosis."""

    symptoms: list[str] = Field(description="List of patient symptoms")
    patient_context: str = Field(
        default="", description="Age, gender, medical history, etc."
    )


# ── Transcription Segments ──────────────────────────────────────────


class SegmentOut(BaseModel):
    """A single transcription segment with timestamps."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")


# ── Response Models ─────────────────────────────────────────────────


class TranscriptionResponse(BaseModel):
    """Response from the /transcribe endpoint."""

    transcript: str = Field(description="Full concatenated transcript")
    detected_language: str = Field(description="Language code detected by Whisper")
    language_probability: float = Field(
        description="Confidence of language detection (0-1)"
    )
    audio_quality: dict = Field(
        default_factory=dict,
        description="Input audio checks and normalization metadata",
    )
    segments: list[SegmentOut] = Field(
        default_factory=list,
        description="Individual transcript segments with timestamps",
    )


class AnalysisResponse(BaseModel):
    """Response from the /analyze endpoint."""

    transcript: str = Field(description="Full concatenated transcript")
    detected_language: str = Field(description="Language code detected by Whisper")
    audio_quality: dict = Field(
        default_factory=dict,
        description="Input audio checks and normalization metadata",
    )
    extraction: MedicalExtraction = Field(
        description="Structured medical info extracted by Qwen3"
    )


class ExtractResponse(BaseModel):
    """Response from the /extract endpoint."""

    extraction: MedicalExtraction = Field(
        description="Structured medical info extracted by Qwen3"
    )
    model: str = Field(description="Ollama model used")
    mode: str = Field(default="no_think", description="Thinking mode used")


class DiagnoseResponse(BaseModel):
    """Response from the /diagnose endpoint."""

    diagnosis: DiagnosisResult = Field(description="Diagnostic reasoning output")
    model: str = Field(description="Ollama model used")
    mode: str = Field(default="think", description="Thinking mode used")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(default="ok")
    ollama: str = Field(description="Status of Ollama service (ok/down/unknown)")
    model_found: bool = Field(description="Whether the LLM model is installed")
    model_loaded: bool = Field(description="Whether the LLM model is currently in memory")
    whisper_model_loaded: bool = Field(
        description="Whether Whisper model is in memory (local)"
    )
    colab_online: bool = Field(default=False, description="Whether remote Colab GPU is reachable")
    whisper_mode: str = Field(default="local_cpu", description="Current transcription mode (remote_gpu/local_cpu)")


# ── Streaming Models ────────────────────────────────────────────────


class MedicalEntity(BaseModel):
    """A single normalized medical entity extracted by the rule engine."""

    text: str = Field(description="Normalized medical term")
    original: str = Field(description="Original text from transcript")
    category: str = Field(description="Entity category: symptom / condition / medication")
    timestamp: float = Field(default=0.0, description="Detection timestamp (epoch)")


class RedFlagAlert(BaseModel):
    """An emergency / urgent red flag triggered by the rule engine."""

    description: str = Field(description="Human-readable flag description")
    urgency: str = Field(description="emergency / urgent")
    pattern_matched: str = Field(default="", description="The pattern that triggered the flag")


class StreamEvent(BaseModel):
    """A single event pushed over WebSocket to the client."""

    event: str = Field(description="Event type: partial_transcript / medical_entity / red_flag / buffer_update / final_report / error")
    data: dict = Field(default_factory=dict, description="Event payload")
    session_id: str = Field(default="", description="Session ID")
    timestamp: float = Field(default=0.0, description="Event timestamp")


class SessionBufferState(BaseModel):
    """Snapshot of the session buffer sent to the client."""

    session_id: str = Field(description="Session ID")
    symptoms: list[MedicalEntity] = Field(default_factory=list)
    conditions: list[MedicalEntity] = Field(default_factory=list)
    medications: list[MedicalEntity] = Field(default_factory=list)
    red_flags: list[RedFlagAlert] = Field(default_factory=list)
    overall_urgency: str = Field(default="routine")
    confidence_score: float = Field(default=0.0)
    chunks_processed: int = Field(default=0)


class FinalReport(BaseModel):
    """Structured final report from the LLM (post-buffer processing)."""

    chief_complaints: list[str] = Field(default_factory=list)
    patient_reported_symptoms: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    past_conditions_mentioned: list[str] = Field(default_factory=list)
    conditions_mentioned: list[str] = Field(default_factory=list)
    medications_mentioned: list[str] = Field(default_factory=list)
    prescribed_medications: list[MedicineItem] = Field(default_factory=list)
    body_parts_mentioned: list[str] = Field(default_factory=list)
    duration: list[str] = Field(default_factory=list)
    severity: list[str] = Field(default_factory=list)
    doctor_observations: list[str] = Field(default_factory=list)
    doctor_confirmed_diagnosis: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)
    urgency_level: str = Field(default="routine")
    session_id: str = Field(default="")
    model: str = Field(default="")

