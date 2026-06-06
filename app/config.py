"""Application configuration settings."""

import httpx

# ── Colab GPU API ─────────────────────────────────────────────────────────────
# Paste the ngrok base URL printed by Colab Cell 8 here.
# Example: "https://xxxx.ngrok-free.app" or "https://xxxx.ngrok-free.app/transcribe"
# Leave empty ("") to use local CPU fallback automatically.
WHISPER_API_URL: str = "https://repair-dolly-dollop.ngrok-free.dev/transcribe"
WHISPER_TIMEOUT: int = 300  # 5 min max wait

# ── Local Whisper fallback (used only when WHISPER_API_URL is empty) ──────────
WHISPER_MODEL: str = "large-v2"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"  # saves RAM on CPU
DEFAULT_TRANSCRIPTION_LANGUAGE: str = "ml"
FORCE_TRANSCRIPTION_LANGUAGE: bool = True
WHISPER_INITIAL_PROMPT: str = (
    "Malayalam doctor patient medical consultation. "
    "Common Manglish words: pani fever, thalavedana headache, nenju vedana chest pain, "
    "swasam muttal breathing difficulty, chumma cough, sugar diabetes, BP blood pressure, "
    "ksheenam tiredness, thalakarakkam dizziness. "
    "Malayalam terms: പനി, തലവേദന, നെഞ്ചുവേദന, ശ്വാസം മുട്ടൽ, ചുമ, ഷുഗർ, ബി പി, ക്ഷീണം, തലകറക്കം."
)

# ── Ollama / Qwen3 Settings ────────────────────────────────────────────────────
# Set OLLAMA_API_URL to the Colab ngrok base + "/ollama" to use the GPU proxy.
# Example: "https://xxxx.ngrok-free.app/ollama"
# Leave empty ("") to fall back to a local Ollama instance on localhost:11434.
OLLAMA_API_URL: str = "https://repair-dolly-dollop.ngrok-free.dev/ollama"

OLLAMA_BASE_URL: str = "http://localhost:11434"   # local fallback
OLLAMA_MODEL: str = "qwen3:8b"
OLLAMA_TIMEOUT: float = 600.0  # seconds — generous for first-token latency on GPU

# Derived URLs — automatically use Colab proxy when OLLAMA_API_URL is set.
_ollama_effective_base: str = (
    OLLAMA_API_URL.rstrip("/") if OLLAMA_API_URL else OLLAMA_BASE_URL.rstrip("/")
)
OLLAMA_CHAT_URL: str = f"{_ollama_effective_base}/api/chat"
OLLAMA_GENERATE_URL: str = f"{_ollama_effective_base}/api/generate"
OLLAMA_TAGS_URL: str = f"{_ollama_effective_base}/api/tags"
OLLAMA_PS_URL: str = f"{_ollama_effective_base}/api/ps"

# httpx.Timeout object for use in AsyncClient — properly sets read timeout.
OLLAMA_HTTPX_TIMEOUT: httpx.Timeout = httpx.Timeout(OLLAMA_TIMEOUT)

# ── VAD (Voice Activity Detection) Settings ───────────────────────────────────
VAD_FILTER: bool = True
VAD_MIN_SILENCE_MS: int = 500
BEAM_SIZE: int = 5

# ── Streaming Settings ────────────────────────────────────────────────────────
STREAM_CHUNK_DURATION_SEC: float = 3.0        # expected audio chunk length
STREAM_MAX_SESSION_DURATION_SEC: float = 1800.0  # 30 min max session
BUFFER_CONFIDENCE_THRESHOLD: float = 0.7      # auto-finalize when reached
