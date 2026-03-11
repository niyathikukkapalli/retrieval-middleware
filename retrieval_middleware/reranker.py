import time
from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL, RERANKER_TOP_N


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = RERANKER_TOP_N,
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not documents:
            return [], 0.0

        pairs = [(query, doc["text"]) for doc in documents]

        start = time.perf_counter()
        scores = self._model.predict(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        scored_docs: List[Dict[str, Any]] = []
        for doc, score in zip(documents, scores):
            scored_docs.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )

        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        top_n = min(top_n, len(scored_docs))
        return scored_docs[:top_n], elapsed_ms

