"""Tests for POST /v1/chat/completions endpoint."""

from __future__ import annotations

import json

import pytest


def _make_body(model="test-model", stream=False, **kwargs):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "max_tokens": 100,
        **kwargs,
    }


# ── Non-streaming tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_non_stream_returns_200(client):
    resp = await client.post("/v1/chat/completions", json=_make_body())
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_non_stream_response_structure(client):
    data = (await client.post("/v1/chat/completions", json=_make_body())).json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "test-model"

    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "stop"

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_non_stream_generates_expected_text(client):
    data = (await client.post("/v1/chat/completions", json=_make_body())).json()
    content = data["choices"][0]["message"]["content"]
    assert "你好" in content


# ── Streaming tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_returns_event_stream(client):
    resp = await client.post(
        "/v1/chat/completions", json=_make_body(stream=True)
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_contains_done_marker(client):
    resp = await client.post(
        "/v1/chat/completions", json=_make_body(stream=True)
    )
    text = resp.text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_stream_chunks_are_valid_json(client):
    resp = await client.post(
        "/v1/chat/completions", json=_make_body(stream=True)
    )
    chunks = _parse_sse(resp.text)
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")
        assert len(chunk["choices"]) == 1


@pytest.mark.asyncio
async def test_stream_last_chunk_has_usage(client):
    resp = await client.post(
        "/v1/chat/completions", json=_make_body(stream=True)
    )
    chunks = _parse_sse(resp.text)
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"] is not None
    assert last["usage"]["prompt_tokens"] > 0
    assert last["usage"]["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_stream_content_matches_non_stream(client):
    non_stream = (await client.post(
        "/v1/chat/completions", json=_make_body()
    )).json()

    stream_resp = await client.post(
        "/v1/chat/completions", json=_make_body(stream=True)
    )
    chunks = _parse_sse(stream_resp.text)
    stream_text = ""
    for c in chunks:
        delta = c["choices"][0]["delta"].get("content")
        if delta:
            stream_text += delta

    assert stream_text == non_stream["choices"][0]["message"]["content"]


# ── Error handling tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_model_returns_404(client):
    resp = await client.post(
        "/v1/chat/completions", json=_make_body(model="nonexistent")
    )
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_missing_messages_returns_422(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-model"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_temperature_returns_422(client):
    resp = await client.post(
        "/v1/chat/completions",
        json=_make_body(temperature=5.0),
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_system_message_accepted(client):
    body = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 50,
    }
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200


# ── Helper ────────────────────────────────────────────────────

def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of JSON chunk dicts."""
    chunks = []
    for line in text.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            data = line[6:]
            try:
                chunks.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return chunks
