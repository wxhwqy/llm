"""Common schemas: errors, health check, etc."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class QueueStatus(BaseModel):
    active: int
    waiting: int
    max_concurrent: int


class HealthResponse(BaseModel):
    status: str
    models: dict
    queue: QueueStatus
    uptime_seconds: float
