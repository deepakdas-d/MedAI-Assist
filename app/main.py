"""MedAssist — FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import audio, health, streaming

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown logic."""
    logger.info("🚀 Starting MedAssist API …")
    yield
    logger.info("👋 MedAssist API shut down")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedAssist API",
    description=(
        "Medical conversation pipeline — Whisper transcription "
        "(Malayalam / English / Manglish) + Qwen3 medical extraction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio.router)
app.include_router(health.router)
app.include_router(streaming.router)
