from retrieval_middleware.reranker import Reranker


def test_reranker_returns_top_n_results():
    reranker = Reranker()
    docs = [
        {"id": str(i), "text": f"Document {i}", "score": 0.0}
        for i in range(10)
    ]

    top_n = 5
    results, elapsed_ms = reranker.rerank("test query", docs, top_n=top_n)

    assert len(results) == top_n
    assert elapsed_ms >= 0.0


def test_results_sorted_by_descending_score():
    reranker = Reranker()
    docs = [
        {"id": str(i), "text": f"Doc {i}", "score": 0.0}
        for i in range(5)
    ]

    results, _ = reranker.rerank("another query", docs, top_n=5)
    scores = [doc["score"] for doc in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_handles_fewer_than_top_n_docs():
    reranker = Reranker()
    docs = [
        {"id": "1", "text": "Only document", "score": 0.0},
    ]

    results, _ = reranker.rerank("query", docs, top_n=5)
    assert len(results) == 1
    assert results[0]["id"] == "1"

