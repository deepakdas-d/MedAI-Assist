"""Real-time streaming endpoints — WebSocket audio + session management.

Implements the core streaming architecture:
    Audio chunks → Whisper → Medical Filter → Buffer → (finalize) → LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import BUFFER_CONFIDENCE_THRESHOLD, OLLAMA_MODEL
from app.services.medical_filter import filter_chunk
from app.services.red_flag_engine import check_red_flags
from app.services.symptom_buffer import SessionBuffer
from app.services.extraction import structure_from_buffer
from app.services.audio_preprocessing import cleanup_prepared_audio, prepare_audio_for_whisper
from app.whisper_client import whisper_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["streaming"])

# In-memory session store (swap for Redis in production)
_active_sessions: dict[str, SessionBuffer] = {}


# ── Helpers ─────────────────────────────────────────────────────────


def _make_event(event: str, data: dict, session_id: str) -> str:
    """Serialize a stream event to JSON for sending over WebSocket."""
    return json.dumps(
        {
            "event": event,
            "data": data,
            "session_id": session_id,
            "timestamp": time.time(),
        }
    )


async def _safe_send(ws: WebSocket, text: str) -> bool:
    """Send text over WebSocket, returning False if the connection is gone.

    Prevents the double-crash that happens when the client disconnects
    while the server is mid-finalization and we try to send the report
    to a dead socket.
    """
    try:
        await ws.send_text(text)
        return True
    except Exception:
        return False


async def _safe_close(ws: WebSocket) -> None:
    """Close the WebSocket safely — no crash if already closed."""
    try:
        await ws.close()
    except Exception:
        pass


async def _transcribe_chunk(audio_bytes: bytes, chunk_index: int) -> dict:
    """Write audio bytes to a temp file and transcribe via Whisper."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    prepared = None
    try:
        tmp.write(audio_bytes)
        tmp.close()
        prepared = prepare_audio_for_whisper(tmp.name)
        result = await whisper_client.transcribe(prepared.path)
        result["audio_quality"] = prepared.quality
    finally:
        if prepared is not None:
            cleanup_prepared_audio(prepared)
        os.unlink(tmp.name)
    return result


async def _finalize_session(buffer: SessionBuffer) -> dict:
    """Send the full transcript to the LLM for final structuring.

    Uses buffer.is_empty which now checks transcript_chunks — so even
    when the keyword filter found nothing, we still call the LLM if
    there is actual speech in the transcript.
    """
    empty = {
        "chief_complaints": [],
        "patient_reported_symptoms": [],
        "symptoms": [],
        "past_conditions_mentioned": [],
        "conditions_mentioned": [],
        "medications_mentioned": [],
        "prescribed_medications": [],
        "body_parts_mentioned": [],
        "duration": [],
        "severity": [],
        "doctor_observations": [],
        "doctor_confirmed_diagnosis": [],
        "advice": [],
        "recommended_tests": [],
        "follow_up": [],
        "risk_flags": [],
        "uncertain_items": [],
    }

    # ✅ is_empty now checks transcript_chunks, not just keyword matches
    if buffer.is_empty:
        logger.warning("Buffer has no transcript — skipping LLM call")
        return empty

    prompt = buffer.to_llm_prompt()
    logger.info(
        "Finalizing session %s — transcript chunks: %d — prompt length: %d chars\nPROMPT PREVIEW:\n%s",
        buffer.session_id,
        len(buffer.transcript_chunks),
        len(prompt),
        prompt[:600],
    )

    try:
        result = await asyncio.wait_for(
            structure_from_buffer(prompt),
            timeout=120,  # ✅ Reduced from 180s — if Qwen takes >2min something is wrong
        )
        logger.info("✅ LLM returned result: %s", result)
        return result
    except asyncio.TimeoutError:
        logger.error("❌ LLM call timed out after 120s — returning empty report")
        return empty
    except Exception as exc:
        logger.error("❌ LLM call failed: %s", exc)
        return empty


# ── WebSocket Endpoint ──────────────────────────────────────────────


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """Real-time audio streaming endpoint.

    Protocol
    --------
    **Client → Server (binary):** Raw audio chunk bytes (2–5 sec WAV/PCM).

    **Client → Server (text):**
        ``{"action": "finalize"}`` — end the session and get final report.

    **Server → Client (text — JSON events):**
        - ``partial_transcript`` — Whisper output for this chunk
        - ``medical_entity``    — each extracted entity
        - ``red_flag``          — emergency/urgent flag
        - ``buffer_update``     — current buffer snapshot
        - ``final_report``      — structured LLM output
        - ``error``             — processing error
    """
    await ws.accept()
    buffer = SessionBuffer()
    _active_sessions[buffer.session_id] = buffer
    chunk_index = 0

    logger.info("🔌 WebSocket session started: %s", buffer.session_id)

    # Send session start event
    await ws.send_text(
        _make_event("session_start", {"session_id": buffer.session_id}, buffer.session_id)
    )

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("type") != "websocket.receive":
                continue

            # ── Text message (control commands) ─────────────────────
            if message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await _safe_send(
                        ws,
                        _make_event("error", {"message": "Invalid JSON"}, buffer.session_id),
                    )
                    continue

                action = payload.get("action", "")

                if action == "finalize":
                    logger.info(
                        "📋 Finalizing session %s — buffer state: %s",
                        buffer.session_id,
                        buffer.to_dict(),
                    )

                    # ✅ Always clean up session from store first
                    _active_sessions.pop(buffer.session_id, None)

                    report_data = await _finalize_session(buffer)
                    logger.info("✅ Got report_data: %s", report_data)

                    final = {
                        "chief_complaints": report_data.get("chief_complaints", []),
                        "patient_reported_symptoms": report_data.get(
                            "patient_reported_symptoms", report_data.get("symptoms", [])
                        ),
                        "symptoms": report_data.get("symptoms", []),
                        "past_conditions_mentioned": report_data.get(
                            "past_conditions_mentioned",
                            report_data.get("conditions_mentioned", []),
                        ),
                        "conditions_mentioned": report_data.get("conditions_mentioned", []),
                        "medications_mentioned": report_data.get("medications_mentioned", []),
                        "prescribed_medications": report_data.get("prescribed_medications", []),
                        "body_parts_mentioned": report_data.get("body_parts_mentioned", []),
                        "duration": report_data.get("duration", []),
                        "severity": report_data.get("severity", []),
                        "doctor_observations": report_data.get("doctor_observations", []),
                        "doctor_confirmed_diagnosis": report_data.get(
                            "doctor_confirmed_diagnosis", []
                        ),
                        "advice": report_data.get("advice", []),
                        "recommended_tests": report_data.get("recommended_tests", []),
                        "follow_up": report_data.get("follow_up", []),
                        "risk_flags": report_data.get("risk_flags", []),
                        "uncertain_items": report_data.get("uncertain_items", []),
                        "urgency_level": buffer.overall_urgency,
                        "session_id": buffer.session_id,
                        "model": OLLAMA_MODEL,
                    }

                    # ✅ _safe_send: no crash if client disconnected while LLM was running
                    sent = await _safe_send(
                        ws,
                        _make_event("final_report", final, buffer.session_id),
                    )
                    if not sent:
                        logger.warning(
                            "Client disconnected before final_report could be sent — session %s",
                            buffer.session_id,
                        )

                    await _safe_close(ws)
                    return

                elif action == "get_buffer":
                    await _safe_send(
                        ws,
                        _make_event("buffer_update", buffer.to_dict(), buffer.session_id),
                    )

                continue

            # ── Binary message (audio chunk) ────────────────────────
            if message.get("bytes"):
                audio_bytes: bytes = message["bytes"]
                chunk_index += 1
                now = time.time()

                # 1. Transcribe the chunk
                try:
                    result = await _transcribe_chunk(audio_bytes, chunk_index)
                except Exception as exc:
                    logger.error("Transcription failed for chunk %d: %s", chunk_index, exc)
                    await _safe_send(
                        ws,
                        _make_event(
                            "error",
                            {"message": f"Transcription failed: {exc}", "chunk": chunk_index},
                            buffer.session_id,
                        ),
                    )
                    continue

                transcript = result.get("transcript", result.get("text", ""))
                if not transcript.strip():
                    continue  # silence / empty chunk

                # Send partial transcript
                await _safe_send(
                    ws,
                    _make_event(
                        "partial_transcript",
                        {
                            "text": transcript,
                            "chunk": chunk_index,
                            "audio_quality": result.get("audio_quality", {}),
                        },
                        buffer.session_id,
                    ),
                )

                # 2. Medical filter (rule-based, <10ms)
                filter_result = filter_chunk(transcript)

                # 3. Red flag check (<5ms)
                red_flag_result = check_red_flags(transcript)

                # 4. Update buffer
                buffer.ingest_filter_result(filter_result, now, chunk_index)
                buffer.add_red_flags(red_flag_result)
                buffer.add_transcript_chunk(transcript)

                # 5. Send extracted entities
                for entity in filter_result.symptoms + filter_result.conditions + filter_result.medications:
                    await _safe_send(
                        ws,
                        _make_event(
                            "medical_entity",
                            {
                                "text": entity.normalized,
                                "original": entity.original,
                                "category": entity.category,
                            },
                            buffer.session_id,
                        ),
                    )

                # 6. Send red flags immediately
                if red_flag_result.has_flags:
                    for flag in red_flag_result.flags:
                        await _safe_send(
                            ws,
                            _make_event(
                                "red_flag",
                                {
                                    "description": flag.description,
                                    "urgency": flag.urgency,
                                    "pattern": flag.pattern_matched,
                                },
                                buffer.session_id,
                            ),
                        )

                # 7. Send buffer snapshot
                await _safe_send(
                    ws,
                    _make_event("buffer_update", buffer.to_dict(), buffer.session_id),
                )

                # 8. Auto-finalize hint if confidence threshold reached
                if buffer.confidence_score >= BUFFER_CONFIDENCE_THRESHOLD:
                    logger.info(
                        "📊 Confidence %.2f ≥ threshold %.2f — auto-finalize hint sent",
                        buffer.confidence_score,
                        BUFFER_CONFIDENCE_THRESHOLD,
                    )
                    await _safe_send(
                        ws,
                        _make_event(
                            "confidence_threshold",
                            {
                                "score": round(buffer.confidence_score, 3),
                                "message": "Buffer confidence threshold reached. "
                                'Send {"action": "finalize"} when ready.',
                            },
                            buffer.session_id,
                        ),
                    )

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected: %s", buffer.session_id)
        _active_sessions.pop(buffer.session_id, None)
    except Exception as exc:
        logger.exception("WebSocket error in session %s", buffer.session_id)
        _active_sessions.pop(buffer.session_id, None)


# ── REST Endpoints ──────────────────────────────────────────────────


@router.get("/stream/sessions")
async def list_sessions() -> dict:
    """List active streaming sessions."""
    return {
        "active_sessions": [
            {
                "session_id": sid,
                "chunks_processed": buf.chunks_processed,
                "confidence_score": round(buf.confidence_score, 3),
                "urgency": buf.overall_urgency,
                "symptoms_count": len(buf.symptoms),
            }
            for sid, buf in _active_sessions.items()
        ]
    }


@router.get("/stream/session/{session_id}/buffer")
async def get_session_buffer(session_id: str) -> dict:
    """Get the current buffer state for a session."""
    buf = _active_sessions.get(session_id)
    if not buf:
        return {"error": "Session not found", "session_id": session_id}
    return buf.to_dict()
