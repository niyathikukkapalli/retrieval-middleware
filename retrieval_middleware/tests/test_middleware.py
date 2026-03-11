import pytest
from httpx import AsyncClient

from retrieval_middleware.main import app, cache


@pytest.mark.asyncio
async def test_cold_query_end_to_end():
    cache.clear()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
    assert response.status_code == 200
    data = response.json()

    assert data["cache_hit"] is False
    assert len(data["results"]) == 5

    latency = data["latency_breakdown"]
    assert latency["embedding_ms"] >= 0.0
    assert latency["cache_lookup_ms"] >= 0.0
    assert latency["vector_db_ms"] >= 0.0
    assert latency["reranking_ms"] >= 0.0
    assert latency["total_ms"] >= 0.0


@pytest.mark.asyncio
async def test_warm_query_cache_hit():
    cache.clear()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        first = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
        assert first.status_code == 200

        second = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
    assert second.status_code == 200
    data = second.json()

    assert data["cache_hit"] is True
    latency = data["latency_breakdown"]
    assert latency["vector_db_ms"] == 0.0
    assert latency["reranking_ms"] == 0.0


@pytest.mark.asyncio
async def test_near_duplicate_query_cache_hit():
    cache.clear()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        first = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
        assert first.status_code == 200

        second = await ac.post("/query", json={"query": "Define machine learning", "top_n": 5})
    assert second.status_code == 200
    data = second.json()

    assert data["cache_hit"] is True


@pytest.mark.asyncio
async def test_cache_delete_resets_behavior():
    cache.clear()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        first = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
        assert first.status_code == 200

        delete_resp = await ac.delete("/cache")
        assert delete_resp.status_code == 200

        second = await ac.post("/query", json={"query": "What is machine learning?", "top_n": 5})
    assert second.status_code == 200
    data = second.json()

    assert data["cache_hit"] is False

