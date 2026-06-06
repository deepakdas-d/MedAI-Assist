# GOOGLE COLAB UPDATE CELLS
#
# Run these in your Colab notebook before starting uvicorn.
# Recommended order:
#   1. Install/load your normal FastAPI + ngrok setup cells.
#   2. Run CELL A below to register the improved Whisper /transcribe endpoint.
#   3. Run CELL B and CELL C for Ollama.
#   4. Run CELL D before uvicorn starts so /ollama/{path} is registered.
#   5. Start uvicorn/ngrok and paste the printed URLs into local config.py.


# ---------------------------------------------------------------------------
# CELL A - Improved Whisper endpoint for Malayalam-English medical audio
# ---------------------------------------------------------------------------
import os
import subprocess
import tempfile
import time

from fastapi import File, Form, UploadFile
from faster_whisper import WhisperModel


WHISPER_MODEL_NAME = "large-v2"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "ml"
DEFAULT_INITIAL_PROMPT = (
    "Malayalam-English doctor patient medical consultation. "
    "Transcribe Malayalam speech in Malayalam script and preserve English words, "
    "medicine names, doses, numbers, BP, sugar, tablet, injection, morning, night, "
    "after food, fever, cough, headache, chest pain, breathing difficulty, dizziness."
)

print(f"Loading Whisper {WHISPER_MODEL_NAME} on GPU...")
whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("Whisper model loaded")


def _bool_form(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


def _convert_to_whisper_wav(input_path: str) -> str:
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output.close()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output.name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        try:
            os.unlink(output.name)
        except FileNotFoundError:
            pass
        raise RuntimeError(f"ffmpeg conversion failed: {completed.stderr[-1000:]}")
    return output.name


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(DEFAULT_LANGUAGE),
    task: str = Form("transcribe"),
    initial_prompt: str | None = Form(None),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
    condition_on_previous_text: str = Form("false"),
    vad_filter: str = Form("true"),
    vad_min_silence_ms: int = Form(500),
    no_speech_threshold: float = Form(0.65),
    log_prob_threshold: float = Form(-1.0),
    compression_ratio_threshold: float = Form(2.4),
):
    input_path = _save_upload_to_temp(file)
    wav_path = None
    started = time.time()
    try:
        wav_path = _convert_to_whisper_wav(input_path)
        effective_language = None if language == "auto" else (language or DEFAULT_LANGUAGE)
        segments_iter, info = whisper_model.transcribe(
            wav_path,
            language=effective_language,
            task=task,
            initial_prompt=initial_prompt or DEFAULT_INITIAL_PROMPT,
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=_bool_form(condition_on_previous_text, False),
            vad_filter=_bool_form(vad_filter, True),
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )

        segments = []
        transcript_parts = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            transcript_parts.append(text)
            segments.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                    "avg_logprob": round(getattr(segment, "avg_logprob", 0.0), 3),
                    "no_speech_prob": round(getattr(segment, "no_speech_prob", 0.0), 3),
                    "compression_ratio": round(
                        getattr(segment, "compression_ratio", 0.0), 3
                    ),
                }
            )

        transcript = " ".join(transcript_parts).strip()
        print(f"Transcribed {file.filename} in {time.time() - started:.2f}s: {transcript}")
        return {
            "success": True,
            "transcript": transcript,
            "text": transcript,
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }
    finally:
        for path in (input_path, wav_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


print("Improved Whisper route registered: POST /transcribe")


# ---------------------------------------------------------------------------
# CELL B - Install and start Ollama inside Colab
# ---------------------------------------------------------------------------
import requests


print("Installing Ollama...")
install = subprocess.run(
    "curl -fsSL https://ollama.com/install.sh | sh",
    shell=True,
    capture_output=True,
    text=True,
)
if install.returncode != 0:
    print("STDERR:", install.stderr[-2000:])
    raise RuntimeError("Ollama installation failed")
print("Ollama binary installed")

print("Starting Ollama server...")
ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

for attempt in range(30):
    time.sleep(2)
    try:
        r = requests.get("http://localhost:11434/", timeout=3)
        if r.status_code == 200:
            print(f"Ollama is running after {(attempt + 1) * 2}s")
            break
    except Exception:
        pass
else:
    raise RuntimeError("Ollama did not start within 60 seconds")


# ---------------------------------------------------------------------------
# CELL C - Pull Qwen3 model
# ---------------------------------------------------------------------------
QWEN_MODEL = "qwen3:8b"

print(f"Pulling {QWEN_MODEL}. This can take several minutes on first run...")
pull = subprocess.run(["ollama", "pull", QWEN_MODEL], capture_output=False)
if pull.returncode != 0:
    raise RuntimeError(f"Failed to pull {QWEN_MODEL}")

tags = requests.get("http://localhost:11434/api/tags").json()
model_names = [m["name"] for m in tags.get("models", [])]
if not any(QWEN_MODEL.split(":")[0] in name for name in model_names):
    raise RuntimeError(f"{QWEN_MODEL} not found after pull. Available: {model_names}")

print(f"{QWEN_MODEL} is ready. Available models: {model_names}")


# ---------------------------------------------------------------------------
# CELL D - Ollama reverse proxy route
# ---------------------------------------------------------------------------
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse


@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    target_url = f"http://localhost:11434/{path}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )

    return StreamingResponse(
        content=iter([resp.content]),
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


print("Ollama proxy route registered: /ollama/{path}")

try:
    print()
    print("=" * 65)
    print("BOTH APIs LIVE - update your local config.py / .env:")
    print("=" * 65)
    print(f'WHISPER_API_URL = "{public_url}/transcribe"')
    print(f'OLLAMA_API_URL  = "{public_url}/ollama"')
    print("=" * 65)
    print(f"Smoke-test Whisper: {public_url}/")
    print(f"Smoke-test Ollama:  {public_url}/ollama/api/tags")
    print("=" * 65)
except NameError:
    print("public_url is not defined yet. Print the URLs after ngrok starts.")
