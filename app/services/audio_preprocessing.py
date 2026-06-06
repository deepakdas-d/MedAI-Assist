"""Audio preparation helpers for Whisper input.

The prototype needs stable Malayalam transcription, so uploaded WAV files are
normalized to the shape Whisper handles best: 16 kHz, mono, signed 16-bit PCM.
Non-WAV files are left untouched and passed through to the remote API.
"""

from __future__ import annotations

import audioop
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


@dataclass
class PreparedAudio:
    """Prepared audio path plus lightweight quality metadata."""

    path: str
    temp_paths: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)


def _read_wav(path: str) -> tuple[bytes, dict] | None:
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            frames = wf.readframes(frame_count)
    except wave.Error:
        return None

    duration = frame_count / sample_rate if sample_rate else 0.0
    metadata = {
        "input_format": "wav",
        "input_sample_rate": sample_rate,
        "input_channels": channels,
        "input_sample_width": sample_width,
        "duration_sec": round(duration, 2),
    }
    return frames, metadata


def _quality_metrics(frames: bytes, sample_width: int) -> dict:
    if not frames:
        return {"rms": 0, "peak": 0, "is_silent": True, "is_clipping": False}

    rms = audioop.rms(frames, sample_width)
    peak = audioop.max(frames, sample_width)
    max_value = float((1 << (8 * sample_width - 1)) - 1)
    return {
        "rms": int(rms),
        "peak": int(peak),
        "is_silent": rms < 100,
        "is_clipping": peak >= int(max_value * 0.98),
    }


def _normalize_wav_frames(
    frames: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> tuple[bytes, bool]:
    changed = False

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
        changed = True

    if sample_width != TARGET_SAMPLE_WIDTH:
        frames = audioop.lin2lin(frames, sample_width, TARGET_SAMPLE_WIDTH)
        sample_width = TARGET_SAMPLE_WIDTH
        changed = True

    if sample_rate != TARGET_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            TARGET_CHANNELS,
            sample_rate,
            TARGET_SAMPLE_RATE,
            None,
        )
        changed = True

    return frames, changed


def _write_wav(frames: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_SAMPLE_WIDTH)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(frames)
    return tmp.name


def _convert_with_ffmpeg(path: str) -> PreparedAudio | None:
    """Convert any readable audio container to Whisper-friendly PCM WAV."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path.close()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        path,
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        "pcm_s16le",
        output_path.name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        try:
            os.unlink(output_path.name)
        except FileNotFoundError:
            pass
        return None

    wav = _read_wav(output_path.name)
    quality = {
        "input_format": os.path.splitext(path)[1].lstrip(".") or "unknown",
        "normalized": True,
        "conversion": "ffmpeg",
    }
    if wav is not None:
        frames, metadata = wav
        quality.update(metadata)
        quality.update(_quality_metrics(frames, TARGET_SAMPLE_WIDTH))
        quality.update(
            {
                "output_sample_rate": TARGET_SAMPLE_RATE,
                "output_channels": TARGET_CHANNELS,
                "output_sample_width": TARGET_SAMPLE_WIDTH,
            }
        )

    return PreparedAudio(path=output_path.name, temp_paths=[output_path.name], quality=quality)


def prepare_audio_for_whisper(path: str) -> PreparedAudio:
    """Normalize WAV audio and return the path Whisper should receive."""

    wav = _read_wav(path)
    if wav is None:
        converted = _convert_with_ffmpeg(path)
        if converted is not None:
            return converted
        return PreparedAudio(
            path=path,
            quality={
                "input_format": os.path.splitext(path)[1].lstrip(".") or "unknown",
                "normalized": False,
                "note": "non-wav input passed through",
            },
        )

    frames, metadata = wav
    normalized, changed = _normalize_wav_frames(
        frames,
        int(metadata["input_sample_rate"]),
        int(metadata["input_channels"]),
        int(metadata["input_sample_width"]),
    )

    quality = {
        **metadata,
        **_quality_metrics(normalized, TARGET_SAMPLE_WIDTH),
        "output_sample_rate": TARGET_SAMPLE_RATE,
        "output_channels": TARGET_CHANNELS,
        "output_sample_width": TARGET_SAMPLE_WIDTH,
        "normalized": changed,
    }

    if not changed:
        return PreparedAudio(path=path, quality=quality)

    normalized_path = _write_wav(normalized)
    return PreparedAudio(path=normalized_path, temp_paths=[normalized_path], quality=quality)


def cleanup_prepared_audio(prepared: PreparedAudio) -> None:
    """Remove temporary files created during preparation."""

    for path in prepared.temp_paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
