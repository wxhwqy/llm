"""Tests for GET /health endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_response_structure(client):
    data = (await client.get("/health")).json()
    assert data["status"] == "healthy"
    assert data["models"]["total"] == 1
    assert data["models"]["loaded"] == 1
    assert data["queue"]["active"] == 0
    assert data["queue"]["waiting"] == 0
    assert data["queue"]["max_concurrent"] == 1
    assert isinstance(data["uptime_seconds"], (int, float))
