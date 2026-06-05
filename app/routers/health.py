import logging
import httpx
from fastapi import APIRouter
from app.config import OLLAMA_TAGS_URL, OLLAMA_PS_URL, OLLAMA_MODEL
from app.schemas import HealthResponse
from app.whisper_client import whisper_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health")
async def health_check() -> HealthResponse:
    """Check the health of the API and its dependencies (Ollama)."""
    health_status = {
        "status": "ok",
        "ollama": "unknown",
        "model_found": False,
        "model_loaded": False,
        "whisper_model_loaded": bool(whisper_client._local_model),
        "colab_online": await whisper_client.health_check(),
        "whisper_mode": "remote_gpu" if whisper_client.use_remote else "local_cpu"
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1. Check if Ollama is responsive and if model exists
            tags_response = await client.get(OLLAMA_TAGS_URL)
            if tags_response.status_code == 200:
                health_status["ollama"] = "ok"
                models = tags_response.json().get("models", [])
                # Normalize model name check
                target_model = OLLAMA_MODEL.split(":")[0]
                health_status["model_found"] = any(
                    target_model in m.get("name", "") for m in models
                )
            else:
                health_status["ollama"] = f"error_{tags_response.status_code}"
                health_status["status"] = "degraded"

            # 2. Check if model is currently loaded in memory
            if health_status["ollama"] == "ok":
                ps_response = await client.get(OLLAMA_PS_URL)
                if ps_response.status_code == 200:
                    loaded_models = ps_response.json().get("models", [])
                    # The name in 'ps' might be slightly different or same as tags
                    health_status["model_loaded"] = any(
                        target_model in m.get("name", "") for m in loaded_models
                    )
                
    except httpx.RequestError as exc:
        logger.error(f"Error checking Ollama health: {exc}")
        health_status["ollama"] = "down"
        health_status["status"] = "degraded"
    
    return HealthResponse(**health_status)
