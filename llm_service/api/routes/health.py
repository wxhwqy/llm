"""GET /health — service health check."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from api.schemas.common import HealthResponse, QueueStatus
from api.services.model_manager import ModelManager
from api.services.queue import InferenceQueue

router = APIRouter()

_start_time = time.monotonic()


def _get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def _get_queue(request: Request) -> InferenceQueue:
    return request.app.state.queue


@router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: ModelManager = Depends(_get_model_manager),
    queue: InferenceQueue = Depends(_get_queue),
):
    models_info = manager.list_models()
    loaded_count = sum(1 for m in models_info if m["status"] == "loaded")
    q = queue.stats

    return HealthResponse(
        status="healthy" if loaded_count > 0 else "degraded",
        models={"total": len(models_info), "loaded": loaded_count},
        queue=QueueStatus(
            active=q.active,
            waiting=q.waiting,
            max_concurrent=q.max_concurrent,
        ),
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )
