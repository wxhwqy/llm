"""Pydantic schemas for OpenAI-compatible chat completion API."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    stop: list[str] | None = None
    no_think: bool = Field(default=False, description="Disable Qwen3 thinking mode (appends /no_think)")


# ── Response (non-streaming) ─────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice]
    usage: UsageInfo


# ── Response (streaming chunks) ──────────────────────────────

class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None
