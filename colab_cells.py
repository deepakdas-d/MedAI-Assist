# ╔══════════════════════════════════════════════════════════════════╗
# ║  GOOGLE COLAB — CELLS 6, 7, 8                                   ║
# ║  Add these cells AFTER your existing Cell 5 (ngrok tunnel cell)  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# ✅ CELL 6 — Install Ollama inside the Colab runtime
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
import time
import requests

print('Installing Ollama...')
install = subprocess.run(
    'curl -fsSL https://ollama.com/install.sh | sh',
    shell=True, capture_output=True, text=True
)
if install.returncode != 0:
    print('STDERR:', install.stderr[-2000:])
    raise RuntimeError('Ollama installation failed')
print('✅ Ollama binary installed!')

# Start the Ollama server in the background
print('Starting Ollama server...')
ollama_proc = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait until Ollama responds on port 11434
for attempt in range(30):
    time.sleep(2)
    try:
        r = requests.get('http://localhost:11434/', timeout=3)
        if r.status_code == 200:
            print(f'✅ Ollama is running! (ready after {(attempt+1)*2}s)')
            break
    except Exception:
        pass
else:
    raise RuntimeError('Ollama did not start within 60 seconds')


# ─────────────────────────────────────────────────────────────────────────────
# ✅ CELL 7 — Pull the Qwen3 model
# ─────────────────────────────────────────────────────────────────────────────
# qwen3:8b  ≈ 5 GB  — best quality
# qwen3:1.7b ≈ 1 GB — faster pull, lower quality (swap if Colab disk is tight)
QWEN_MODEL = 'qwen3:8b'

print(f'Pulling {QWEN_MODEL} — this can take several minutes on first run...')
pull = subprocess.run(
    ['ollama', 'pull', QWEN_MODEL],
    capture_output=False   # show progress in real time
)
if pull.returncode != 0:
    raise RuntimeError(f'Failed to pull {QWEN_MODEL}')

# Verify the model is listed
tags = requests.get('http://localhost:11434/api/tags').json()
model_names = [m['name'] for m in tags.get('models', [])]
if not any(QWEN_MODEL.split(':')[0] in name for name in model_names):
    raise RuntimeError(f'{QWEN_MODEL} not found after pull. Available: {model_names}')

print(f'✅ {QWEN_MODEL} is ready! Available models: {model_names}')


# ─────────────────────────────────────────────────────────────────────────────
# ✅ CELL 8 — Add /ollama proxy route to FastAPI & print final URLs
# ─────────────────────────────────────────────────────────────────────────────
# This must run BEFORE uvicorn.run() in Cell 5.
# Move this cell above Cell 5 (the uvicorn / ngrok cell), or restart the
# runtime and run the cells in order: 1→2→3→4→8→5
#
# If you already ran Cell 5 (server is live), restart the runtime and re-run
# all cells in the correct order (1, 2, 3, 4, 6, 7, 8, 5).
# ─────────────────────────────────────────────────────────────────────────────

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

@app.api_route('/ollama/{path:path}', methods=['GET', 'POST', 'DELETE'])
async def ollama_proxy(path: str, request: Request):
    """
    Transparent reverse proxy → http://localhost:11434/{path}

    Lets the local FastAPI app call:
        POST <ngrok-url>/ollama/api/chat
    which forwards to Ollama running inside this Colab instance.
    """
    target_url = f'http://localhost:11434/{path}'
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ('host', 'content-length')
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

print('✅ Ollama proxy route registered: /ollama/{path}')

# ── Print the final URLs for your local config.py ──────────────────────────
# public_url was set in Cell 5 by ngrok.connect(8000)
# Make sure this cell runs AFTER the ngrok tunnel is created.
try:
    print()
    print('=' * 65)
    print('🚀  BOTH APIs LIVE — update your local config.py / .env:')
    print('=' * 65)
    print()
    print(f'  WHISPER_API_URL = "{public_url}/transcribe"')
    print(f'  OLLAMA_API_URL  = "{public_url}/ollama"')
    print()
    print('=' * 65)
    print(f'  Smoke-test Whisper:  {public_url}/')
    print(f'  Smoke-test Ollama:   {public_url}/ollama/api/tags')
    print('=' * 65)
    print()
    print('⚠️  Paste both URLs into your local app/config.py, then restart')
    print('    the local FastAPI server.')
    print('⚠️  The ngrok URL changes each time you restart this notebook!')
except NameError:
    print('⚠️  public_url not defined yet — run Cell 5 first to start ngrok.')
