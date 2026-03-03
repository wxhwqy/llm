"""Tests for InferenceQueue concurrency control."""

from __future__ import annotations

import asyncio

import pytest

from api.services.queue import InferenceQueue, QueueFullError, QueueTimeoutError


@pytest.mark.asyncio
async def test_acquire_release_basic():
    q = InferenceQueue(max_concurrent=1, max_queue_size=4, timeout=2.0)
    assert q.stats.active == 0
    async with q.acquire():
        assert q.stats.active == 1
    assert q.stats.active == 0


@pytest.mark.asyncio
async def test_concurrent_requests_are_serialized():
    q = InferenceQueue(max_concurrent=1, max_queue_size=4, timeout=5.0)
    order = []

    async def task(name: str, delay: float):
        async with q.acquire():
            order.append(f"{name}_start")
            await asyncio.sleep(delay)
            order.append(f"{name}_end")

    await asyncio.gather(task("a", 0.1), task("b", 0.1))
    # Tasks should be serialized with max_concurrent=1
    assert order[0] == "a_start"
    assert order[1] == "a_end"
    assert order[2] == "b_start"
    assert order[3] == "b_end"


@pytest.mark.asyncio
async def test_queue_full_raises():
    q = InferenceQueue(max_concurrent=1, max_queue_size=1, timeout=5.0)

    async def occupy():
        async with q.acquire():
            await asyncio.sleep(1.0)

    # Start two tasks to fill queue (1 active + 1 waiting = capacity)
    t1 = asyncio.create_task(occupy())
    t2 = asyncio.create_task(occupy())
    await asyncio.sleep(0.05)  # let them start

    # Third should be rejected
    with pytest.raises(QueueFullError):
        async with q.acquire():
            pass

    t1.cancel()
    t2.cancel()
    try:
        await t1
    except (asyncio.CancelledError, Exception):
        pass
    try:
        await t2
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_queue_timeout_raises():
    q = InferenceQueue(max_concurrent=1, max_queue_size=4, timeout=0.1)

    async def occupy():
        async with q.acquire():
            await asyncio.sleep(2.0)

    t = asyncio.create_task(occupy())
    await asyncio.sleep(0.02)

    with pytest.raises(QueueTimeoutError):
        async with q.acquire():
            pass

    t.cancel()
    try:
        await t
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_stats_tracking():
    q = InferenceQueue(max_concurrent=2, max_queue_size=4, timeout=5.0)
    s = q.stats
    assert s.active == 0
    assert s.waiting == 0
    assert s.max_concurrent == 2
