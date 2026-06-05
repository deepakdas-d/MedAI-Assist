import wave, numpy as np

with wave.open(r"C:\Users\deepa\Downloads\test1.wav", "rb") as f:
    frames = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    rate = f.getframerate()

# Check 1 — is it actually 16kHz mono?
print(f"Sample rate: {rate}")  # should be 16000
print(f"Channels: {f.getnchannels()}")  # should be 1

# Check 2 — is the volume reasonable?
rms = np.sqrt(np.mean(frames.astype(np.float32)**2))
print(f"RMS volume: {rms:.0f}")  # good speech = 800-8000, silence = <100