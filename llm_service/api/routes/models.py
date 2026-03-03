"""GET /v1/models — list available models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.schemas.models import ModelInfo, ModelListResponse
from api.services.model_manager import ModelManager

router = APIRouter()


def _get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    manager: ModelManager = Depends(_get_model_manager),
):
    items = manager.list_models()
    return ModelListResponse(
        data=[
            ModelInfo(
                id=m["id"],
                name=m["name"],
                max_context_length=m["max_context_length"],
                status=m["status"],
            )
            for m in items
        ]
    )
