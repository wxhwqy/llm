"""Core inference service: bridges sync stream_generate to async SSE."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
from uuid import uuid4

from api.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    CompletionChoice,
    UsageInfo,
)
from api.services.model_manager import ModelManager, LoadedModel
from api.services.queue import InferenceQueue

logger = logging.getLogger(__name__)

STOP_TOKENS = {"<|im_end|>", "<|endoftext|>"}

_SENTINEL = None


class ContextLengthExceededError(Exception):
    def __init__(self, prompt_tokens: int, max_seq_len: int):
        self.prompt_tokens = prompt_tokens
        self.max_seq_len = max_seq_len
        super().__init__(
            f"Prompt too long: {prompt_tokens} tokens "
            f"(max sequence length {max_seq_len})"
        )


@dataclass
class _ActiveRequest:
    request_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)


class IncrementalDecoder:
    """Handles BPE tokens that span UTF-8 character boundaries."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._offset = 0

    def add(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._tokenizer.decode(self._ids, skip_special_tokens=False)
        new = text[self._offset:]
        # Only emit complete UTF-8 characters
        try:
            new.encode("utf-8")
        except UnicodeEncodeError:
            return ""
        self._offset = len(text)
        return new


class InferenceService:
    def __init__(self, model_manager: ModelManager, queue: InferenceQueue):
        self._model_manager = model_manager
        self._queue = queue
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")
        self._active: dict[str, _ActiveRequest] = {}

    # ── public API ────────────────────────────────────────────

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        loaded = await self._model_manager.get_model(request.model)
        input_ids, prompt_tokens = self._tokenize(loaded, request)
        max_new = self._resolve_max_tokens(request, prompt_tokens, loaded)

        async with self._queue.acquire():
            async for chunk in self._generate(
                loaded, input_ids, prompt_tokens, max_new, request, request_id
            ):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> ChatCompletionResponse:
        content = ""
        usage = None
        finish_reason = "stop"
        async for chunk in self.chat_completion_stream(request, request_id):
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if chunk.usage:
                usage = chunk.usage

        if usage is None:
            usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return ChatCompletionResponse(
            id=request_id,
            model=request.model,
            choices=[CompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )],
            usage=usage,
        )

    def cancel(self, request_id: str) -> bool:
        req = self._active.get(request_id)
        if req:
            req.cancelled.set()
            return True
        return False

    # ── internals ─────────────────────────────────────────────

    def _tokenize(
        self, loaded: LoadedModel, request: ChatCompletionRequest
    ) -> tuple[list[int], int]:
        messages = [m.model_dump() for m in request.messages]
        text = loaded.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = loaded.tokenizer.encode(text)
        return ids, len(ids)

    @staticmethod
    def _resolve_max_tokens(
        request: ChatCompletionRequest,
        prompt_tokens: int,
        loaded: LoadedModel,
    ) -> int:
        max_new = request.max_tokens or 2048
        remaining = loaded.config.max_seq_len - prompt_tokens
        if remaining <= 0:
            raise ContextLengthExceededError(
                prompt_tokens, loaded.config.max_seq_len
            )
        return min(max_new, remaining)

    async def _generate(
        self,
        loaded: LoadedModel,
        input_ids: list[int],
        prompt_tokens: int,
        max_new_tokens: int,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        token_q: asyncio.Queue[int | None] = asyncio.Queue()
        active = _ActiveRequest(request_id=request_id)
        self._active[request_id] = active
        loop = asyncio.get_running_loop()

        future = loop.run_in_executor(
            self._executor,
            self._inference_thread,
            loaded.model,
            input_ids,
            max_new_tokens,
            request.temperature,
            request.top_k,
            request.top_p,
            token_q,
            active.cancelled,
            loop,
        )

        decoder = IncrementalDecoder(loaded.tokenizer)
        completion_tokens = 0
        t0 = time.monotonic()
        ttft: float | None = None
        finish_reason = "stop"

        try:
            while True:
                token_id = await token_q.get()
                if token_id is _SENTINEL:
                    break

                completion_tokens += 1
                if ttft is None:
                    ttft = time.monotonic() - t0

                text = decoder.add(token_id)
                if not text:
                    continue

                # Check for stop tokens in the decoded text
                should_stop = False
                for st in STOP_TOKENS:
                    if st in text:
                        text = text.split(st)[0]
                        should_stop = True
                        break

                if text:
                    yield ChatCompletionChunk(
                        id=request_id,
                        model=request.model,
                        choices=[ChunkChoice(delta=ChunkDelta(content=text))],
                    )

                if should_stop:
                    break

            # Final chunk with finish_reason and usage
            elapsed = time.monotonic() - t0
            tps = completion_tokens / elapsed if elapsed > 0 else 0

            logger.info(
                "Inference complete: id=%s prompt=%d completion=%d "
                "ttft=%.0fms total=%.1fs tps=%.1f",
                request_id, prompt_tokens, completion_tokens,
                (ttft or 0) * 1000, elapsed, tps,
            )

            yield ChatCompletionChunk(
                id=request_id,
                model=request.model,
                choices=[ChunkChoice(
                    delta=ChunkDelta(),
                    finish_reason=finish_reason,
                )],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except asyncio.CancelledError:
            active.cancelled.set()
            raise
        finally:
            self._active.pop(request_id, None)
            # Wait for inference thread to finish
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

    @staticmethod
    def _inference_thread(
        model: Any,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        token_q: asyncio.Queue,
        cancelled: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Runs in a dedicated thread — calls the synchronous engine."""
        try:
            for token_id in model.stream_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                if cancelled.is_set():
                    break
                asyncio.run_coroutine_threadsafe(
                    token_q.put(token_id), loop
                ).result(timeout=10)
        except Exception:
            logger.exception("Inference thread error")
        finally:
            asyncio.run_coroutine_threadsafe(
                token_q.put(_SENTINEL), loop
            ).result(timeout=10)
