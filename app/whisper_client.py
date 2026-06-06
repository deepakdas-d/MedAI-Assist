import httpx
import os
import asyncio
from pathlib import Path
import logging

from app.config import (
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    FORCE_TRANSCRIPTION_LANGUAGE,
    WHISPER_API_URL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_MODEL,
    WHISPER_TIMEOUT,
)

logger = logging.getLogger(__name__)

class WhisperClient:
    """
    Unified Whisper client.
    - If WHISPER_API_URL is set → uses Colab GPU API (fast)
    - If not set               → falls back to local CPU (slow but works)
    """

    def __init__(self, api_url: str = WHISPER_API_URL):
        # Strip trailing slashes and the /transcribe suffix if user pasted the full endpoint
        url = api_url.strip().rstrip("/") if api_url else ""
        if url.endswith("/transcribe"):
            url = url[:-11].rstrip("/")
        
        self.api_url = url
        self.use_remote = bool(self.api_url)
        self._local_model = None  # lazy-loaded only if needed

    def _effective_language(self, language: str | None) -> str | None:
        """Return the language to send to Whisper.

        The prototype defaults to Malayalam because auto-detect has been
        unstable on generated/demo audio. Passing language="auto" disables
        the forced language for comparison tests.
        """
        if language and language.lower() == "auto":
            return None
        if language:
            return language
        if FORCE_TRANSCRIPTION_LANGUAGE:
            return DEFAULT_TRANSCRIPTION_LANGUAGE
        return None

    # ── Remote (Colab GPU) ────────────────────────────────────────────────
    async def transcribe_remote(
        self,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict:
        """Send audio to Colab GPU API and return transcript."""
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        async with httpx.AsyncClient(timeout=WHISPER_TIMEOUT) as client:
            with open(audio_path, "rb") as f:
                import mimetypes
                mime_type, _ = mimetypes.guess_type(audio_path)
                mime_type = mime_type or "application/octet-stream"
                files = {"file": (audio_file.name, f, mime_type)}
                data = {"task": task}
                effective_language = self._effective_language(language)
                if effective_language:
                    data["language"] = effective_language
                if WHISPER_INITIAL_PROMPT:
                    data["initial_prompt"] = WHISPER_INITIAL_PROMPT

                response = await client.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                )

        response.raise_for_status()
        return response.json()

    # ── Local fallback (CPU) ──────────────────────────────────────────────
    def _load_local_model(self):
        """Lazy-load local Whisper model only when needed."""
        if self._local_model is None:
            # We import here to avoid dependency if not used
            from faster_whisper import WhisperModel
            logger.info(f"Loading local Whisper {WHISPER_MODEL} model (CPU, {WHISPER_COMPUTE_TYPE})...")
            self._local_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
            logger.info("✅ Local Whisper model loaded.")
        return self._local_model

    def transcribe_local(
        self,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict:
        """Transcribe locally using CPU (fallback)."""
        model = self._load_local_model()
        effective_language = self._effective_language(language)
        segments, info = model.transcribe(
            audio_path,
            language=effective_language,
            task=task,
            beam_size=5,
            vad_filter=True,
            initial_prompt=WHISPER_INITIAL_PROMPT or None,
        )
        result_segments = []
        full_text = ""
        for seg in segments:
            result_segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip(),
            })
            full_text += seg.text

        return {
            "success": True,
            "text": full_text.strip(),
            "segments": result_segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }

    # ── Unified entry point ───────────────────────────────────────────────
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict:
        """
        Auto-selects remote GPU or local CPU.
        Returns a dict compatible with TranscriptionResponse.
        """
        if self.use_remote:
            logger.info(f"🚀 Using Colab GPU API: {self.api_url}")
            try:
                result = await self.transcribe_remote(audio_path, language, task)
                # The remote API might return 'text' instead of 'transcript' based on the snippet
                if "text" in result and "transcript" not in result:
                    result["transcript"] = result.pop("text")
                return result
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                logger.warning(f"⚠️  Colab API error ({e}). Falling back to local CPU...")
                # auto-fallback to local if Colab is down
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.transcribe_local, audio_path, language, task
                )
        else:
            logger.info("💻 Using local CPU (set WHISPER_API_URL to use Colab GPU)")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.transcribe_local, audio_path, language, task
            )

    async def health_check(self) -> bool:
        """Check if Colab API is reachable."""
        if not self.use_remote:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.api_url}/")
                return r.status_code == 200
        except Exception:
            return False


# Singleton instance
whisper_client = WhisperClient()
