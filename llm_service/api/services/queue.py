"""Inference request queue with concurrency control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator


class QueueFullError(Exception):
    def __init__(self, queue_size: int):
        self.queue_size = queue_size
        super().__init__(f"Queue full ({queue_size} waiting)")


class QueueTimeoutError(Exception):
    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(f"Queue wait timed out after {timeout}s")


@dataclass
class QueueStats:
    active: int
    waiting: int
    max_concurrent: int


class InferenceQueue:
    """
    Concurrency gate for inference requests.

    Uses an asyncio.Semaphore to limit the number of concurrent inferences.
    Phase 1: max_concurrent=1 (engine KV-Cache is single-request).
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        max_queue_size: int = 16,
        timeout: float = 120.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._timeout = timeout
        self._waiting = 0
        self._active = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        if self._waiting >= self._max_queue_size:
            raise QueueFullError(self._waiting)

        self._waiting += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            self._waiting -= 1
            raise QueueTimeoutError(self._timeout)

        self._waiting -= 1
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._semaphore.release()

    @property
    def stats(self) -> QueueStats:
        return QueueStats(
            active=self._active,
            waiting=self._waiting,
            max_concurrent=self._max_concurrent,
        )
