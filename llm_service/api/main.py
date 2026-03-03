"""FastAPI application entry-point for the LLM inference service."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.middleware.error_handler import register_error_handlers
from api.routes import chat, health, models
from api.services.inference import InferenceService
from api.services.model_manager import ModelManager
from api.services.queue import InferenceQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("llm_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info(
        "Starting LLM Service  (models=%d, max_concurrent=%d)",
        len(settings.model_configs),
        settings.max_concurrent,
    )

    model_manager = ModelManager(settings.model_configs)
    queue = InferenceQueue(
        max_concurrent=settings.max_concurrent,
        max_queue_size=settings.max_queue_size,
        timeout=settings.queue_timeout_seconds,
    )
    inference = InferenceService(model_manager, queue)

    await model_manager.startup(settings.default_model)

    app.state.model_manager = model_manager
    app.state.queue = queue
    app.state.inference = inference

    logger.info("LLM Service ready.")
    yield

    logger.info("Shutting down LLM Service …")
    await model_manager.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Service (llaisys)",
        description="OpenAI-compatible inference API backed by the llaisys CUDA engine.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)

    app.include_router(chat.router, tags=["Chat"])
    app.include_router(models.router, tags=["Models"])
    app.include_router(health.router, tags=["Health"])

    return app


app = create_app()
