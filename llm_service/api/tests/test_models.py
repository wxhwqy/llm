"""Tests for GET /v1/models endpoint."""

import pytest


@pytest.mark.asyncio
async def test_list_models_returns_200(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_models_structure(client):
    data = (await client.get("/v1/models")).json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1

    model = data["data"][0]
    assert model["id"] == "test-model"
    assert model["name"] == "Test Model"
    assert model["max_context_length"] == 8192
    assert model["status"] == "loaded"
    assert model["object"] == "model"
    assert model["owned_by"] == "llaisys"
