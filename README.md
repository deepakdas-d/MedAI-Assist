# MedAssist — AI Medical Pipeline

A high-performance medical assistant API that combines **Faster-Whisper** for transcription and **Ollama/Qwen3** for smart medical information extraction with dual-mode thinking logic.

## 🚀 Key Features

- **Multilingual Transcription**: Whisper `large-v2` support for Malayalam, English, and Manglish (code-switching).
- **Smart Thinking Modes**:
  - **`/no_think`**: Fast, structured extraction of symptoms, medicines, and diagnoses.
  - **`/think`**: Deep diagnostic reasoning for vague or complex symptoms.
- **Async API**: Built with FastAPI and `httpx` for high-performance non-blocking I/O.
- **Structured Data**: Returns clean, validated JSON via Pydantic models.

## 🛠 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Configuration**:
   Ensure Ollama is running and has the Qwen3 model:
   ```bash
   ollama run qwen3:8b
   ```

3. **Run API**:
   ```bash
   fastapi dev app/main.py
   ```

## 📡 API Reference

| Endpoint | Method | Description | Model Mode |
|----------|--------|-------------|------------|
| `/api/transcribe` | `POST` | Audio → Transcript | Whisper |
| `/api/analyze` | `POST` | Audio → Data | Whisper + Qwen3 (`no_think`) |
| `/api/extract` | `POST` | Text → Data | Qwen3 (`no_think`) |
| `/api/diagnose` | `POST` | Symptoms → Diagnosis | Qwen3 (`think`) |
| `/api/health` | `GET` | Health Check | System |

Detailed schemas and request bodies can be found at `http://localhost:8000/docs`.

## 📂 Project Structure

- `app/main.py`: FastAPI entry point and model lifespan management.
- `app/routers/`: API route definitions.
- `app/services/`: Core logic for Whisper and Ollama/Qwen3.
- `app/schemas.py`: Pydantic models for structured data validation.
- `app/config.py`: Configuration for models and endpoints.
