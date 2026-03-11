from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, gt=0)


class Document(BaseModel):
    id: str
    text: str
    score: float


class LatencyBreakdown(BaseModel):
    embedding_ms: float
    cache_lookup_ms: float
    vector_db_ms: float
    reranking_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    results: List[Document]
    cache_hit: bool
    cache_similarity_score: float
    latency_breakdown: LatencyBreakdown


class StatsResponse(BaseModel):
    hit_count: int
    miss_count: int
    hit_rate: float
    current_size: int

    @classmethod
    def from_cache_stats(cls, stats: Dict[str, Any]) -> "StatsResponse":
        return cls(
            hit_count=int(stats["hit_count"]),
            miss_count=int(stats["miss_count"]),
            hit_rate=float(stats["hit_rate"]),
            current_size=int(stats["current_size"]),
        )

