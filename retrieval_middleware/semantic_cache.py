from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import CACHE_MAX_SIZE, CACHE_SIMILARITY_THRESHOLD, EMBEDDING_DIM


@dataclass
class CacheEntry:
    embedding: np.ndarray
    results: List[Dict[str, Any]]


class SemanticCache:
    def __init__(
        self,
        max_size: int = CACHE_MAX_SIZE,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
    ) -> None:
        self._max_size = max_size
        self._similarity_threshold = similarity_threshold
        self._entries: Deque[CacheEntry] = deque()
        self._hit_count = 0
        self._miss_count = 0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape[0] != EMBEDDING_DIM or b.shape[0] != EMBEDDING_DIM:
            raise ValueError("Unexpected embedding dimension")
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def get(self, query_embedding: np.ndarray) -> Tuple[bool, Optional[List[Dict[str, Any]]], float]:
        best_similarity = 0.0
        best_results: Optional[List[Dict[str, Any]]] = None

        for entry in self._entries:
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_results = entry.results

        if best_results is not None and best_similarity >= self._similarity_threshold:
            self._hit_count += 1
            return True, best_results, best_similarity

        self._miss_count += 1
        return False, None, best_similarity

    def set(self, query_embedding: np.ndarray, results: List[Dict[str, Any]]) -> None:
        if len(self._entries) >= self._max_size:
            self._entries.popleft()
        self._entries.append(CacheEntry(embedding=query_embedding.copy(), results=results))

    def stats(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100.0) if total > 0 else 0.0
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "current_size": len(self._entries),
        }

    def clear(self) -> None:
        self._entries.clear()
        self._hit_count = 0
        self._miss_count = 0

