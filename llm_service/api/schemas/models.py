"""Pydantic schemas for the /v1/models endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llaisys"
    name: str = ""
    max_context_length: int = 16384
    status: str = "available"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
