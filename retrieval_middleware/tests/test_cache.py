import numpy as np
import pytest
from fastapi.testclient import TestClient

from retrieval_middleware.config import EMBEDDING_DIM
from retrieval_middleware.main import app, cache
from retrieval_middleware.semantic_cache import SemanticCache


client = TestClient(app)


def test_identical_query_cache_hit_similarity_one():
    semantic_cache = SemanticCache()
    embedding = np.ones(EMBEDDING_DIM, dtype=float)
    results = [{"id": "1", "text": "doc", "score": 1.0}]

    semantic_cache.set(embedding, results)

    hit, cached_results, similarity = semantic_cache.get(embedding)
    assert hit is True
    assert cached_results == results
    assert pytest.approx(similarity, rel=1e-6) == 1.0


def test_semantically_different_query_cache_miss():
    semantic_cache = SemanticCache()
    base_embedding = np.ones(EMBEDDING_DIM, dtype=float)
    semantic_cache.set(base_embedding, [{"id": "1", "text": "doc", "score": 1.0}])

    different_embedding = np.zeros(EMBEDDING_DIM, dtype=float)
    hit, cached_results, similarity = semantic_cache.get(different_embedding)

    assert hit is False
    assert cached_results is None
    assert similarity == 0.0


def test_cache_eviction_fifo():
    max_size = 3
    semantic_cache = SemanticCache(max_size=max_size)

    for i in range(max_size + 1):
        emb = np.full(EMBEDDING_DIM, float(i))
        semantic_cache.set(emb, [{"id": str(i), "text": f"doc {i}", "score": 1.0}])

    assert semantic_cache.stats()["current_size"] == max_size


def test_stats_endpoint_reflects_hit_miss_counts():
    cache.clear()

    # Trigger one miss via API
    response = client.post("/query", json={"query": "What is AI?", "top_n": 1})
    assert response.status_code == 200

    # Trigger hit with same query
    response = client.post("/query", json={"query": "What is AI?", "top_n": 1})
    assert response.status_code == 200

    stats_response = client.get("/stats")
    assert stats_response.status_code == 200
    data = stats_response.json()

    assert data["hit_count"] >= 1
    assert data["miss_count"] >= 1
    assert data["current_size"] >= 1

