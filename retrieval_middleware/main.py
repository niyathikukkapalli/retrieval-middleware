import time
from typing import List

import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RERANKER_TOP_N,
    VECTOR_DB_TOP_K,
)
from .models import QueryRequest, QueryResponse, Document, StatsResponse, LatencyBreakdown
from .semantic_cache import SemanticCache
from .vector_db_client import VectorDBClient
from .reranker import Reranker


app = FastAPI(title="Retrieval Middleware POC")

# Lazy-loaded so server binds immediately; models load on first /query
_embedding_model = None
_reranker = None
vector_client = VectorDBClient()
cache = SemanticCache()


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _embed_query(query: str) -> np.ndarray:
    embedding = _get_embedding_model().encode(query, convert_to_numpy=True)
    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {embedding.shape[0]}")
    return embedding


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    total_start = time.perf_counter()

    # 1. Embed query
    t0 = time.perf_counter()
    embedding = _embed_query(request.query)
    embedding_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Cache lookup
    t0 = time.perf_counter()
    hit, cached_results, similarity_score = cache.get(embedding)
    cache_lookup_ms = (time.perf_counter() - t0) * 1000.0

    vector_db_ms = 0.0
    reranking_ms = 0.0
    results: List[Document] = []

    if hit and cached_results is not None:
        # Cache hit: skip vector DB and reranking
        results = [Document(**doc) for doc in cached_results[: request.top_n]]
    else:
        # 4. Vector DB query
        t0 = time.perf_counter()
        candidates = await vector_client.query(embedding, top_k=VECTOR_DB_TOP_K)
        vector_db_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Re-rank
        t0 = time.perf_counter()
        reranked, reranking_ms = _get_reranker().rerank(request.query, candidates, top_n=request.top_n or RERANKER_TOP_N)
        results = [Document(**doc) for doc in reranked]

        # 6. Store in cache
        cache.set(embedding, [doc.model_dump() for doc in results])

    total_ms = (time.perf_counter() - total_start) * 1000.0

    latency = LatencyBreakdown(
        embedding_ms=embedding_ms,
        cache_lookup_ms=cache_lookup_ms,
        vector_db_ms=vector_db_ms,
        reranking_ms=reranking_ms,
        total_ms=total_ms,
    )

    return QueryResponse(
        results=results,
        cache_hit=hit,
        cache_similarity_score=similarity_score,
        latency_breakdown=latency,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    return StatsResponse.from_cache_stats(cache.stats())


@app.delete("/cache")
async def clear_cache_endpoint() -> dict:
    cache.clear()
    return {"status": "ok"}

