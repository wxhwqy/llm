"""Global exception handlers for the FastAPI application."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.services.model_manager import ModelNotFoundError, ModelLoadError
from api.services.queue import QueueFullError, QueueTimeoutError
from api.services.inference import ContextLengthExceededError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": {
                "message": f"Model '{exc.model_id}' not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }},
        )

    @app.exception_handler(ContextLengthExceededError)
    async def _context_length(request: Request, exc: ContextLengthExceededError):
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }},
        )

    @app.exception_handler(QueueFullError)
    async def _queue_full(request: Request, exc: QueueFullError):
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": "Server busy, please retry later",
                "type": "server_error",
                "code": "queue_full",
            }},
        )

    @app.exception_handler(QueueTimeoutError)
    async def _queue_timeout(request: Request, exc: QueueTimeoutError):
        return JSONResponse(
            status_code=504,
            content={"error": {
                "message": f"Queue wait timed out ({exc.timeout}s)",
                "type": "server_error",
                "code": "queue_timeout",
            }},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {
                "message": "Internal server error",
                "type": "server_error",
            }},
        )
