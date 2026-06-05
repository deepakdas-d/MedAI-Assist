# MedAssist Project Analysis Report

## 1. Purpose and Workflow

The project is trying to convert a doctor-patient consultation into a digital medical report.

The intended flow is:

1. Doctor and patient talk naturally.
2. Audio is transcribed.
3. Casual/non-medical conversation is ignored.
4. Medical details are extracted: symptoms, conditions mentioned, medicines, body parts, duration, severity, and risk flags.
5. The doctor reviews and confirms the final report.

Important product rule: the AI should assist extraction and reporting. It should not become the final diagnosing authority. Any diagnosis or treatment decision should remain doctor-approved.

## 2. Colab Part vs FastAPI Part

| Area | Files / Cells | Role |
| --- | --- | --- |
| Google Colab GPU runtime | Pasted notebook cells 1-9 | Runs heavy AI models on Colab GPU and exposes them through ngrok. |
| Colab Whisper server | Pasted notebook cells 3-5 | Loads `faster-whisper` `large-v2` on CUDA and exposes `/transcribe` and `/transcribe/srt`. |
| Colab Ollama/Qwen server | Pasted notebook cells 6-8 and `colab_cells.py` | Installs Ollama, pulls `qwen3:8b`, and proxies `/ollama/{path}` to Ollama inside Colab. |
| Local FastAPI app | `app/main.py` | Main application entry point; registers routers and CORS. |
| Local audio API | `app/routers/audio.py` | Provides `/api/transcribe`, `/api/analyze`, `/api/extract`, `/api/diagnose`. |
| Local streaming API | `app/routers/streaming.py` | WebSocket endpoint for chunked consultation audio: `/api/ws/stream`. |
| Remote Whisper client | `app/whisper_client.py` | Sends audio to Colab `/transcribe`; falls back to local CPU Whisper if remote is unavailable. |
| Extraction service | `app/services/extraction.py` | Sends prompts to Qwen/Ollama and parses JSON. |
| Rule-based medical filter | `app/services/medical_filter.py`, `medical_lexicon.py`, `red_flag_engine.py`, `symptom_buffer.py` | Fast local filtering, buffering, deduplication, and red flag detection for streaming mode. |
| Schemas | `app/schemas.py` | Defines Pydantic response/request models. |
| Tests / manual utilities | `test_whisper.py`, `test_ws.py`, `make_test.py` | Manual smoke testing and sample audio generation. Not a real automated accuracy test suite yet. |

## 3. Current Architecture

### Batch Path

`/api/analyze` receives an audio file, saves it temporarily, sends it to the Colab Whisper API, receives the transcript, then sends the transcript to Qwen through Ollama for structured extraction.

Code locations:

- `app/routers/audio.py:120`
- `app/whisper_client.py:18`
- `app/services/extraction.py:148`

### Modular Path

The project also supports separate steps:

- `/api/transcribe`: audio to transcript.
- `/api/extract`: transcript to structured medical fields.
- `/api/diagnose`: symptoms to possible diagnoses.

Code locations:

- `app/routers/audio.py:80`
- `app/routers/audio.py:167`
- `app/routers/audio.py:190`

### Streaming Path

The WebSocket endpoint receives audio chunks, transcribes each chunk, runs a local rule-based filter, buffers medical entities, sends partial updates, and finalizes through the LLM when requested.

Code locations:

- `app/routers/streaming.py:133`
- `app/services/symptom_buffer.py:197`

## 4. Efficiency Assessment

### What Is Efficient Already

- Heavy Whisper transcription is moved to Colab GPU, which is much faster than local CPU.
- The local app uses async `httpx`, which is good for I/O-heavy calls.
- Whisper uses VAD filtering and beam search in the Colab notebook.
- Local fallback exists if Colab is offline.
- Streaming mode avoids calling the LLM on every small chunk; it uses a fast lexicon/rule layer first.

### Main Inefficiencies

1. Hard-coded ngrok URLs in `app/config.py`.
   - `WHISPER_API_URL` and `OLLAMA_API_URL` are fixed to one temporary Colab URL.
   - Colab/ngrok URLs change often, so this should come from `.env` or environment variables.

2. Colab setup is not production-stable.
   - The notebook installs dependencies, starts services, pulls models, and creates tunnels manually.
   - This is okay for experiments, but it is fragile for real clinic use.

3. Per-chunk streaming transcription is expensive.
   - Each audio chunk is written to a temp file and sent through HTTP/ngrok.
   - This adds disk I/O, network latency, and repeated Whisper overhead.
   - Better: use fewer/larger chunks with overlap, or a true streaming ASR service.

4. Local CPU fallback with `large-v2` is likely too slow.
   - `WHISPER_MODEL` is `large-v2`, device is CPU, compute type is `int8`.
   - It may work, but it will be slow for consultation-length audio.

5. Ollama calls are full-response, non-streaming.
   - `stream: False` is used in `app/services/extraction.py:97`.
   - This is simpler, but it hides progress and increases perceived latency.

6. The final streaming prompt sends the full raw transcript.
   - `SessionBuffer.to_llm_prompt()` sends the whole conversation, not only medically relevant spans.
   - This improves recall, but increases token cost and can reintroduce casual talk.

## 5. Structure Assessment

### Good Structure

- The code is separated into routers, services, schemas, and config.
- `WhisperClient` centralizes remote GPU vs local CPU logic.
- Pydantic schemas give a clean API contract.
- The streaming architecture has a useful buffer design.

### Structural Problems

1. Colab code and app code are mixed conceptually.
   - Colab should be treated as a separate AI runtime service.
   - FastAPI should be the local/backend orchestration service.

2. Configuration is not environment-based.
   - URLs, model names, timeouts, and runtime modes should be read from environment variables.

3. The AI diagnosis endpoint conflicts with the stated workflow.
   - Your requirement says the doctor diagnoses and AI assists extraction.
   - `/api/diagnose` currently asks Qwen to generate differential diagnoses.
   - It should be clearly marked as doctor-assist only, or disabled if the product should only extract report fields.

4. Test files are manual scripts.
   - There is no automated test suite measuring transcription or extraction quality.
   - This makes accuracy improvements difficult to prove.

5. Virtual environments and generated files are inside the project.
   - `med/`, `tts_env/`, `__pycache__/`, and similar generated files should not be part of source control.

6. Encoding appears damaged in several files.
   - Many comments and Malayalam strings appear as mojibake, such as `âœ…` and corrupted Malayalam text.
   - This can affect lexicon matching and test data quality.

## 6. Transcription Accuracy Improvements

Highest-impact changes:

1. Improve audio preprocessing.
   - Always convert to 16 kHz, mono, 16-bit PCM.
   - Normalize volume.
   - Remove background noise when possible.
   - Detect silence and clipping before transcription.

2. Add speaker diarization or speaker labels.
   - The report should know what the patient said versus what the doctor said.
   - Example: symptoms should mainly come from the patient; medicines/treatment may come from the doctor.

3. Keep forced Malayalam for the current prototype demo.
   - Current code defaults to `ml` in both Colab and local Whisper.
   - This is a valid workaround because the generated/demo Malayalam audio is being misclassified as Tamil or multilingual gibberish when language auto-detection is used.
   - For demo reliability, forced Malayalam is better than unstable auto-detection.
   - Later, compare forced `ml` against auto-detect only after using real recorded consultation audio, not only generated TTS demo audio.

4. Use medical vocabulary hints where supported.
   - Add common local medical terms, medicine names, and clinic-specific words as prompt/hotword context.

5. Use chunk overlap in streaming.
   - Audio chunks should overlap slightly, for example 0.5-1 second, to avoid cutting medical words at boundaries.

6. Track confidence per segment.
   - Use Whisper segment confidence/no-speech probabilities where available.
   - Low-confidence spans should be shown to the doctor for review.

7. Build a Malayalam/Manglish test set.
   - Use real-style consultation samples with expected transcripts.
   - Measure WER/CER and entity-level recall.

## 7. Extraction Accuracy Improvements

Highest-impact changes:

1. Add evidence spans.
   - Each extracted item should include the transcript phrase and timestamp that caused it.
   - This helps the doctor verify quickly.

2. Add speaker-aware extraction.
   - Separate patient complaints from doctor instructions.
   - Example fields:
     - `patient_reported_symptoms`
     - `doctor_prescribed_medications`
     - `doctor_advice`
     - `follow_up_plan`

3. Add negation handling.
   - Current extraction can misread "no fever" as "fever".
   - Must detect `no`, `not`, `illa`, `illai`, etc.

4. Expand the schema.
   - `MedicineItem` exists but is not used in `MedicalExtraction`.
   - Medications should include name, dose, frequency, duration, route, and instructions.

5. Separate mentioned conditions from confirmed diagnosis.
   - `conditions_mentioned` should not mean doctor diagnosis.
   - Add a field like `doctor_confirmed_diagnosis`, but only fill it if explicitly spoken by the doctor.

6. Use strict JSON validation and retry.
   - If Qwen returns invalid JSON or wrong fields, automatically retry with the validation error.

7. Add field-level evaluation.
   - Measure precision/recall/F1 for symptoms, medicines, duration, severity, and risk flags.
   - This is more important than only checking if the JSON "looks right".

8. Improve the lexicon.
   - Add more Malayalam Unicode, Manglish spellings, common medicine brand names, dosage phrases, and local symptom phrases.
   - Fix corrupted Malayalam strings first.

## 8. Prototype Security Notes

For the current prototype, speed, transcription accuracy, and extraction accuracy are higher priority than production security. These notes matter later if the project is used with real patient data.

1. A real ngrok auth token appears in the pasted Colab notebook.
   - Rotate that token immediately.
   - Never store auth tokens in notebook text or source files.

2. Consultation audio is sent to Colab through ngrok.
   - This may not be acceptable for real patient data unless privacy, consent, and compliance requirements are handled.

3. CORS is open to all origins.
   - `allow_origins=["*"]` is fine for development, but not production.

4. No authentication is visible.
   - Medical transcription/report endpoints should require authentication.

## 9. Updated Prototype Plan

The new plan should focus on the real prototype goal: fast and accurate medical report generation from Malayalam/Manglish doctor-patient consultation audio. Security and production deployment can wait until the core AI flow is reliable.

### Phase 1: Stabilize Transcription for Demo Accuracy

Goal: make Whisper produce consistent Malayalam transcripts instead of Tamil/gibberish output.

- Keep forced Malayalam: `language="ml"`.
- Treat forced Malayalam as the default demo mode, not as a bug.
- Standardize all input audio before transcription:
  - 16 kHz sample rate.
  - Mono channel.
  - 16-bit PCM WAV.
  - Reasonable volume range.
  - No long silence at the start.
- Add an audio quality check before sending to Whisper:
  - sample rate check.
  - channel count check.
  - RMS volume check.
  - duration check.
- Test Whisper with 3 audio types:
  - generated Malayalam TTS demo audio.
  - manually recorded Malayalam audio.
  - mixed Malayalam-English/Manglish audio.

Success criteria:

- Demo audio should no longer become Tamil/gibberish.
- Transcript should preserve the main medical words: symptoms, medicines, duration, and body parts.

### Phase 2: Make Extraction-Only Report Generation Clear

Goal: AI should extract report details, not independently diagnose.

- Make `/api/analyze` the main prototype endpoint.
- Keep `/api/transcribe` for debugging transcription.
- Keep `/api/extract` for testing extraction from corrected transcript text.
- Move `/api/diagnose` out of the main demo flow or label it as optional doctor-assist reasoning.
- Update prompts to say:
  - extract only explicitly mentioned information.
  - do not infer diagnosis.
  - do not invent symptoms.
  - separate doctor-spoken treatment from patient-spoken complaints.

Success criteria:

- Output behaves like a digital report assistant.
- The doctor remains responsible for final diagnosis.

### Phase 3: Improve Extraction Schema

Goal: make the report useful for a doctor, not just a list of keywords.

Replace the current simple extraction fields with a more practical report structure:

- `chief_complaints`
- `patient_reported_symptoms`
- `duration`
- `severity`
- `body_parts_mentioned`
- `past_conditions_mentioned`
- `doctor_observations`
- `doctor_confirmed_diagnosis`
- `prescribed_medications`
- `advice`
- `recommended_tests`
- `follow_up`
- `red_flags`
- `uncertain_items`

Medication extraction should use structured objects:

- medicine name.
- dosage.
- frequency.
- duration.
- timing, such as before food or after food.

Success criteria:

- The final JSON looks close to a real clinic report.
- Missing or unclear values are empty instead of hallucinated.

### Phase 4: Use Hardcoded Malayalam Phrases Only as a Speed Layer

Goal: avoid depending fully on hardcoded phrase matching.

- Do not hardcode every Malayalam phrase.
- Keep a small lexicon only for fast detection of common terms:
  - fever.
  - headache.
  - cough.
  - chest pain.
  - breathing difficulty.
  - sugar/diabetes.
  - BP/hypertension.
  - common medicine names.
- Let Qwen handle full-context extraction from the transcript.
- Use the lexicon for:
  - quick streaming hints.
  - red flag alerts.
  - fallback matching when LLM output misses obvious terms.

Success criteria:

- The system works even when the exact Malayalam phrase is not hardcoded.
- Hardcoded terms improve speed and recall but do not control the whole report.

### Phase 5: Optimize Speed

Goal: reduce time from audio upload to final report.

- Keep Whisper on Colab GPU.
- Keep Qwen in `/no_think` mode for extraction.
- Avoid `/think` mode in the main report flow.
- Keep prompts short and schema-specific.
- For batch audio, prefer one complete transcription followed by one extraction call.
- For streaming, use chunking only when live updates are required.
- Increase chunk size or add chunk overlap to avoid repeated tiny Whisper calls.

Success criteria:

- Demo report generation should feel quick enough for a consultation workflow.
- Extraction should use one LLM call where possible.

### Phase 6: Add Accuracy Measurement

Goal: stop guessing whether changes improve accuracy.

- Create 10-20 small prototype test cases first.
- Each test case should include:
  - audio file.
  - expected transcript highlights.
  - expected symptoms.
  - expected medicines.
  - expected duration.
  - expected advice/tests.
- Measure:
  - transcription correctness for medical words.
  - symptom extraction recall.
  - medication extraction recall.
  - hallucination count.
  - invalid JSON count.

Success criteria:

- Each model/prompt/audio change can be compared against previous output.
- Accuracy improvements become visible and repeatable.

### Phase 7: Add Review-Friendly Output

Goal: make the report easy for the doctor to verify.

- Add evidence text for each extracted item.
- Add timestamps when available.
- Add `uncertain_items` for low-confidence or unclear transcript parts.
- Keep final output editable by the doctor.

Success criteria:

- Doctor can quickly see why the AI extracted each item.
- Unclear transcript sections are not silently converted into confident report fields.

### Phase 8: Later Production Cleanup

These are not the immediate prototype priority, but should be handled before real use:

- Move ngrok URLs and model settings into `.env`.
- Rotate exposed tokens.
- Add authentication.
- Restrict CORS.
- Remove virtual environments and generated caches from the project folder.
- Replace Colab/ngrok with a more stable hosted inference service if needed.

## 10. Overall Judgment

This is a good prototype architecture: Colab handles heavy AI inference, while local FastAPI orchestrates transcription, extraction, and report generation. The immediate focus should be accurate Malayalam transcription, fast `/no_think` extraction, and a doctor-reviewable report format.

For this stage, the best next improvement is not production security or deployment cleanup. It is a controlled prototype loop: force Malayalam, improve audio preprocessing, extract only explicit medical facts, and measure accuracy with a small labelled demo set.
