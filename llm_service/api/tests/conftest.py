"""Shared fixtures for API tests.

Uses a mock model that yields pre-determined tokens instead of running
real GPU inference, so the full HTTP ↔ SSE pipeline can be exercised
without a GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.config import ModelConfig
from api.main import create_app
from api.services.inference import InferenceService
from api.services.model_manager import ModelManager, LoadedModel
from api.services.queue import InferenceQueue


# ── Fake Qwen3 model ─────────────────────────────────────────

FAKE_TOKENS = [
    29871,  # placeholder ids – the tokenizer mock maps them to text
    29872,
    29873,
]

FAKE_DECODED_TEXT = ["你", "好", "！"]


class FakeQwen3:
    """Drop-in replacement for llaisys.models.Qwen3 in tests."""

    def __init__(self, tokens: list[int] | None = None):
        self.tokens = tokens or FAKE_TOKENS
        self.eos_token_id = 151643
        self.max_seq_len = 8192
        self._reset_count = 0

    def reset(self):
        self._reset_count += 1

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        self.reset()
        for tid in self.tokens:
            yield tid


class FakeTokenizer:
    """Minimal tokenizer mock that works with the inference service."""

    def __init__(self, decode_map: dict[int, str] | None = None):
        self._decode_map = decode_map or dict(zip(FAKE_TOKENS, FAKE_DECODED_TEXT))
        self._call_count = 0

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        return "<fake-prompt>"

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4, 5]  # 5 fake prompt tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        # Incremental decoder calls decode with cumulative ids
        result = ""
        for tid in ids:
            result += self._decode_map.get(tid, "")
        return result


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def model_config():
    return ModelConfig(
        id="test-model",
        name="Test Model",
        path="/tmp/fake-model",
        max_seq_len=8192,
        device="cpu",
        device_ids=[0],
        tp_size=1,
    )


@pytest.fixture
def fake_model():
    return FakeQwen3()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def loaded_model(model_config, fake_model, fake_tokenizer):
    return LoadedModel(
        config=model_config,
        model=fake_model,
        tokenizer=fake_tokenizer,
    )


@pytest.fixture
def model_manager(loaded_model):
    """A ModelManager that already has a model loaded (no GPU needed)."""
    mgr = ModelManager([loaded_model.config])
    mgr._loaded[loaded_model.config.id] = loaded_model
    return mgr


@pytest.fixture
def queue():
    return InferenceQueue(max_concurrent=1, max_queue_size=4, timeout=5.0)


@pytest.fixture
def inference_service(model_manager, queue):
    return InferenceService(model_manager, queue)


@pytest.fixture
def app(model_manager, queue, inference_service):
    """Create a FastAPI app with mocked services injected."""
    application = create_app()
    application.state.model_manager = model_manager
    application.state.queue = queue
    application.state.inference = inference_service
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
