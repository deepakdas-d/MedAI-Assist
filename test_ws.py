import asyncio, websockets, json, wave, io

AUDIO_FILE       = r"C:\Users\deepa\Downloads\test.wav"
WS_URL           = "ws://localhost:8000/api/ws/stream"

SAMPLE_RATE      = 16000
CHANNELS         = 1
BYTES_PER_SAMPLE = 2
CHUNK_SECS       = 4


def pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def print_event(evt: dict):
    """Pretty-print any event with full transcript text visible."""
    name = evt.get("event", "unknown")
    data = evt.get("data", {})

    if name == "partial_transcript":
        text = data.get("text", "").strip()
        chunk = data.get("chunk", "?")
        print(f"[<] partial_transcript  chunk={chunk}")
        print(f"    📝 transcript : {text if text else '(empty)'}")

    elif name == "buffer_update":
        print(f"[<] buffer_update")
        print(f"    symptoms     : {data.get('symptoms', [])}")
        print(f"    conditions   : {data.get('conditions_mentioned', [])}")
        print(f"    medications  : {data.get('medications_mentioned', [])}")
        print(f"    red_flags    : {data.get('red_flags', [])}")
        print(f"    urgency      : {data.get('overall_urgency', '?')}")
        print(f"    confidence   : {data.get('confidence_score', 0)}")
        print(f"    chunks done  : {data.get('chunks_processed', 0)}")

    elif name == "medical_entities":
        print(f"[<] medical_entities")
        print(f"    {json.dumps(data, indent=4, ensure_ascii=False)}")

    elif name == "red_flag":
        print(f"[<] 🚨 red_flag")
        print(f"    {json.dumps(data, indent=4, ensure_ascii=False)}")

    elif name == "final_report":
        print(f"[<] final_report")
        print(f"    {json.dumps(data, indent=2, ensure_ascii=False)}")

    else:
        print(f"[<] {name}")
        if data:
            print(f"    {json.dumps(data, ensure_ascii=False)}")


async def drain_events(ws, timeout=0.3):
    """Read all pending events without blocking. Prints each one."""
    events = []
    try:
        while True:
            raw  = await asyncio.wait_for(ws.recv(), timeout=timeout)
            evt  = json.loads(raw)
            events.append(evt)
            print_event(evt)
    except asyncio.TimeoutError:
        pass
    return events


async def wait_for_event(ws, target_event: str, timeout: int = 60):
    """
    Wait until a specific event type arrives.
    Prints all other events that arrive while waiting.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            print(f"[!] Timed out waiting for '{target_event}'")
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            evt = json.loads(raw)
            print_event(evt)
            if evt.get("event") == target_event:
                return evt
        except asyncio.TimeoutError:
            print(f"[!] Timed out waiting for '{target_event}'")
            return None


async def test():
    print("=" * 60)
    print("  MedAssist WebSocket Test")
    print("=" * 60)

    async with websockets.connect(WS_URL) as ws:

        # ── Step 1: session start ──────────────────────────────────
        msg = json.loads(await ws.recv())
        assert msg["event"] == "session_start", f"Unexpected: {msg}"
        session_id = msg["session_id"]
        print(f"\n[+] Connected   session_id = {session_id}\n")

        # ── Step 2: stream audio chunks ────────────────────────────
        with wave.open(AUDIO_FILE, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            print(f"[i] Audio info  rate={sr}Hz  channels={ch}  "
                  f"frames={wf.getnframes()}  "
                  f"duration={wf.getnframes()/sr:.1f}s\n")

            if sr != SAMPLE_RATE:
                print(f"[!] WARNING: file is {sr}Hz but Whisper expects 16000Hz")
                print(f"    Run: ffmpeg -i test.wav -ar 16000 -ac 1 -sample_fmt s16 test_fixed.wav\n")

            frames_per_chunk = SAMPLE_RATE * CHUNK_SECS
            chunk_num = 0

            while True:
                pcm = wf.readframes(frames_per_chunk)
                if not pcm:
                    break

                chunk_num += 1
                wav_chunk = pcm_to_wav_bytes(pcm)
                await ws.send(wav_chunk)
                print(f"[>] Sent chunk {chunk_num}  ({len(wav_chunk)//1024} KB)")

                # drain any events arriving right after this chunk
                await drain_events(ws, timeout=0.3)

                # ── Step 3: mid-stream buffer check after chunk 3 ──
                if chunk_num == 3:
                    print("\n" + "-" * 40)
                    print("[?] Requesting buffer state...")
                    await ws.send(json.dumps({"action": "get_buffer"}))
                    buf = await wait_for_event(ws, "buffer_update", timeout=10)
                    if buf:
                        d = buf["data"]
                        print(f"\n[buffer @ chunk 3]")
                        print(f"  symptoms    : {d.get('symptoms', [])}")
                        print(f"  conditions  : {d.get('conditions_mentioned', [])}")
                        print(f"  medications : {d.get('medications_mentioned', [])}")
                        print(f"  red_flags   : {d.get('red_flags', [])}")
                        print(f"  urgency     : {d.get('overall_urgency')}")
                        print(f"  confidence  : {d.get('confidence_score')}")
                    print("-" * 40 + "\n")

        # ── Step 4: finalize ───────────────────────────────────────
        print(f"\n[+] All {chunk_num} chunks sent. Finalizing...\n")
        await ws.send(json.dumps({"action": "finalize"}))

        report = await wait_for_event(ws, "final_report", timeout=300)
        print("\n" + "=" * 60)
        if report:
            print("✅  FINAL MEDICAL REPORT")
            print("=" * 60)
            print(json.dumps(report["data"], indent=2, ensure_ascii=False))
        else:
            print("❌  No final_report received.")
            print("    Check streaming.py — finalize handler may not be sending the event.")
        print("=" * 60)


asyncio.run(test())
