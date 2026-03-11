import asyncio
import random
from typing import List, Dict, Any

import numpy as np

from .config import EMBEDDING_DIM, MOCK_DB_LATENCY_MIN_MS, MOCK_DB_LATENCY_MAX_MS, VECTOR_DB_TOP_K


class VectorDBClient:
    def __init__(self, num_documents: int = 256) -> None:
        rng = np.random.default_rng(seed=42)
        # Generate and normalize document embeddings once at startup
        raw_embeddings = rng.normal(size=(num_documents, EMBEDDING_DIM))
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        self._embeddings = raw_embeddings / norms

        self._documents: List[Dict[str, Any]] = []
        for i in range(num_documents):
            self._documents.append(
                {
                    "id": f"doc-{i}",
                    "text": f"Document {i} about topic {i % 10}.",
                }
            )

    async def query(self, embedding: np.ndarray, top_k: int = VECTOR_DB_TOP_K) -> List[Dict[str, Any]]:
        # Simulate network latency
        latency_ms = random.uniform(MOCK_DB_LATENCY_MIN_MS, MOCK_DB_LATENCY_MAX_MS)
        await asyncio.sleep(latency_ms / 1000.0)

        if embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {embedding.shape[0]}")

        # Normalize query embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Zero-norm embedding is not allowed")
        query_vec = embedding / norm

        # Cosine similarity via dot product with normalized vectors
        scores = self._embeddings @ query_vec

        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            doc = self._documents[int(idx)]
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(scores[int(idx)]),
                }
            )
        return results

