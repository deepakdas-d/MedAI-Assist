# make_test.py — generates a proper Malayalam test WAV for Whisper
# Run: python make_test.py
#
# Fixes applied:
#   1. Output path uses os.path — no raw string needed, works on all OS
#   2. ffmpeg call uses check=True so failures are visible, not silent
#   3. temp.mp3 is cleaned up after conversion
#   4. gTTS network errors are caught separately from ffmpeg errors
#   5. Output file existence is verified before declaring success

import os
import subprocess
from gtts import gTTS

# ── Output path ────────────────────────────────────────────────────────────────
# Change this to wherever you want the WAV saved.
# os.path.join works on Windows, Mac, and Linux without raw strings.
OUTPUT_WAV = os.path.join(os.path.expanduser("~"), "Downloads", "test1.wav")
TEMP_MP3   = os.path.join(os.path.dirname(OUTPUT_WAV), "temp_ml.mp3")

# ── Malayalam text ─────────────────────────────────────────────────────────────
# Pure Malayalam Unicode — covers all symptom keywords your filter needs:
#   പനി (fever), തലവേദന (headache), നെഞ്ചുവേദന (chest pain),
#   ശ്വാസം (breathlessness), ഷുഗർ (diabetes / sugar),
#   ബി പി (BP / hypertension), ക്ഷീണം (fatigue), തലകറക്കം (dizziness)
text = (
    "ഡോക്ടർ: എന്താണ് പ്രശ്നം? "
    "രോഗി: എനിക്ക് രണ്ടു ദിവസമായി പനിയുണ്ട്. "
    "തലവേദന വളരെ കൂടുതലാണ്. "
    "നെഞ്ചുവേദനയും ഉണ്ട്. "
    "ശ്വാസം മുട്ടുന്നു. "
    "എനിക്ക് ഷുഗർ ഉണ്ട്. "
    "ബി പി യും ഉണ്ട്. "
    "ക്ഷീണവും തലകറക്കവും ഉണ്ട്."
)

# ── Step 1: gTTS → MP3 ─────────────────────────────────────────────────────────
print("Generating Malayalam audio via gTTS...")
try:
    tts = gTTS(text=text, lang="ml", slow=False)
    tts.save(TEMP_MP3)
    print(f"  MP3 saved: {TEMP_MP3}")
except Exception as e:
    print(f"gTTS failed — check your internet connection.\n  Error: {e}")
    raise SystemExit(1)

# ── Step 2: ffmpeg → 16kHz mono s16 WAV ───────────────────────────────────────
# Whisper requires: 16000 Hz sample rate, mono, 16-bit signed PCM
print("Converting to WAV (16kHz mono s16)...")
try:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", TEMP_MP3,
            "-ar", "16000",   # sample rate Whisper expects
            "-ac", "1",       # mono
            "-sample_fmt", "s16",  # 16-bit signed PCM
            OUTPUT_WAV,
        ],
        check=True,           # raises CalledProcessError if ffmpeg fails
        capture_output=True,  # hides ffmpeg's verbose output unless it errors
    )
except FileNotFoundError:
    print("ffmpeg not found — install it from https://ffmpeg.org/download.html")
    raise SystemExit(1)
except subprocess.CalledProcessError as e:
    print(f"ffmpeg failed:\n{e.stderr.decode()}")
    raise SystemExit(1)
finally:
    # Clean up temp MP3 whether or not conversion succeeded
    if os.path.exists(TEMP_MP3):
        os.remove(TEMP_MP3)
        print(f"  Temp MP3 removed: {TEMP_MP3}")

# ── Step 3: Verify output ──────────────────────────────────────────────────────
if os.path.exists(OUTPUT_WAV):
    size_kb = os.path.getsize(OUTPUT_WAV) / 1024
    print(f"\n  WAV saved : {OUTPUT_WAV}")
    print(f"  File size : {size_kb:.1f} KB")
    print("\nNow upload this WAV to your Whisper endpoint and check the transcript.")
    print("Expected: Malayalam script output (ഡോക്ടർ, പനി, തലവേദന ...)")
    print("If you still see Tamil script (டாக்டார்...), the gTTS bug is still")
    print("occurring — try running the script again or use a different TTS source.")
else:
    print("Output file was not created — check the errors above.")
    raise SystemExit(1)