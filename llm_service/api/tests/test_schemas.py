"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChunkChoice,
    ChunkDelta,
    CompletionChoice,
    UsageInfo,
)


def test_valid_request():
    req = ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.model == "test"
    assert req.stream is False
    assert req.temperature == 0.6
    assert req.max_tokens is None


def test_request_with_all_params():
    req = ChatCompletionRequest(
        model="test",
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ],
        stream=True,
        temperature=1.5,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
        stop=["stop1"],
    )
    assert req.temperature == 1.5
    assert req.max_tokens == 1024
    assert req.stop == ["stop1"]


def test_request_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=3.0,
        )


def test_request_rejects_negative_max_tokens():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=-1,
        )


def test_request_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "invalid", "content": "hi"}],
        )


def test_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="test", messages=[])


def test_chunk_serialization():
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        model="test",
        choices=[ChunkChoice(delta=ChunkDelta(content="hello"))],
    )
    data = chunk.model_dump()
    assert data["choices"][0]["delta"]["content"] == "hello"
    assert data["object"] == "chat.completion.chunk"


def test_response_serialization():
    resp = ChatCompletionResponse(
        id="chatcmpl-test",
        model="test",
        choices=[CompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="hi"),
            finish_reason="stop",
        )],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    data = resp.model_dump()
    assert data["usage"]["total_tokens"] == 15
    assert data["choices"][0]["message"]["content"] == "hi"
