"""POST /v1/chat/completions — OpenAI-compatible chat completion."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from api.services.inference import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"description": "Invalid request (e.g. prompt too long)"},
        404: {"description": "Model not found"},
        503: {"description": "Service busy — queue full"},
        504: {"description": "Queue wait timeout"},
    },
)
async def chat_completions(
    body: ChatCompletionRequest,
    raw_request: Request,
    inference: InferenceService = Depends(_get_inference),
):
    request_id = f"chatcmpl-{uuid4().hex[:24]}"

    if body.stream:
        return StreamingResponse(
            _stream(body, request_id, inference, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await inference.chat_completion(body, request_id)


async def _stream(
    body: ChatCompletionRequest,
    request_id: str,
    inference: InferenceService,
    raw_request: Request,
):
    try:
        async for chunk in inference.chat_completion_stream(body, request_id):
            if await raw_request.is_disconnected():
                logger.info("Client disconnected: %s", request_id)
                inference.cancel(request_id)
                break
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        # Let the global error handler deal with typed exceptions;
        # for SSE we emit an inline error event before closing.
        err = {"error": {"message": str(exc), "type": type(exc).__name__}}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
