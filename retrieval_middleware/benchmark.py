from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .main import query_endpoint, cache
from .models import QueryRequest, QueryResponse


example_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does gradient descent work?",
    "What is reinforcement learning?",
    "Define deep learning",
]


def compare_rankings(cold_results: List[str], cached_results: List[str]) -> Dict[str, Any]:
    if not cold_results and not cached_results:
        return {"top_k_overlap": 1.0, "avg_rank_difference": 0.0, "exact_match": True}

    cold_set = set(cold_results)
    cached_set = set(cached_results)
    union = cold_set | cached_set
    shared = cold_set & cached_set

    top_k_overlap = (len(shared) / len(union)) if union else 0.0

    cold_pos = {doc_id: i for i, doc_id in enumerate(cold_results)}
    cached_pos = {doc_id: i for i, doc_id in enumerate(cached_results)}

    rank_diffs: List[float] = []
    for doc_id in shared:
        rank_diffs.append(abs(cold_pos[doc_id] - cached_pos[doc_id]))

    avg_rank_difference = (sum(rank_diffs) / len(rank_diffs)) if rank_diffs else 0.0
    exact_match = cold_results == cached_results

    return {
        "top_k_overlap": float(top_k_overlap),
        "avg_rank_difference": float(avg_rank_difference),
        "exact_match": bool(exact_match),
    }


def _fmt_ms(ms: float) -> str:
    if math.isnan(ms) or math.isinf(ms):
        return "n/a"
    return f"{ms:.0f} ms"


def _fmt_pct(p: float) -> str:
    if math.isnan(p) or math.isinf(p):
        return "n/a"
    return f"{p:.0f}%"


def _latency_improvement_percent(cold_total: float, cached_total: float) -> float:
    if cold_total <= 0:
        return 0.0
    return max(0.0, (cold_total - cached_total) / cold_total * 100.0)


@dataclass(frozen=True)
class RunResult:
    response: QueryResponse
    doc_ids: List[str]


async def _run_query(query: str, top_n: int) -> RunResult:
    req = QueryRequest(query=query, top_n=top_n)
    resp = await query_endpoint(req)
    doc_ids = [doc.id for doc in resp.results]
    return RunResult(response=resp, doc_ids=doc_ids)


def _print_query_report(
    query: str,
    cold: RunResult,
    cached: RunResult,
    accuracy: Dict[str, Any],
) -> None:
    cold_lat = cold.response.latency_breakdown
    cached_lat = cached.response.latency_breakdown

    speedup_pct = _latency_improvement_percent(cold_lat.total_ms, cached_lat.total_ms)
    vector_db_saved_ms = max(0.0, cold_lat.vector_db_ms - cached_lat.vector_db_ms)
    reranker_saved_ms = max(0.0, cold_lat.reranking_ms - cached_lat.reranking_ms)

    print("-" * 34)
    print(f"QUERY: {query}")
    print("-" * 34)
    print()
    print("Cold Query Latency")
    print(f"embedding: {_fmt_ms(cold_lat.embedding_ms)}")
    print(f"cache lookup: {_fmt_ms(cold_lat.cache_lookup_ms)}")
    print(f"vector DB: {_fmt_ms(cold_lat.vector_db_ms)}")
    print(f"reranking: {_fmt_ms(cold_lat.reranking_ms)}")
    print(f"TOTAL: {_fmt_ms(cold_lat.total_ms)}")
    print()
    print("Cached Query Latency")
    print(f"embedding: {_fmt_ms(cached_lat.embedding_ms)}")
    print(f"cache lookup: {_fmt_ms(cached_lat.cache_lookup_ms)}")
    print(f"vector DB: {_fmt_ms(cached_lat.vector_db_ms)}")
    print(f"reranking: {_fmt_ms(cached_lat.reranking_ms)}")
    print(f"TOTAL: {_fmt_ms(cached_lat.total_ms)}")
    print()
    print("Latency Improvement")
    print(f"{_fmt_pct(speedup_pct)} faster")
    print(f"(vector DB saved: {_fmt_ms(vector_db_saved_ms)}, reranker saved: {_fmt_ms(reranker_saved_ms)})")
    print()
    print("Accuracy Comparison")
    print(f"Top-k overlap: {accuracy['top_k_overlap'] * 100:.0f}%")
    print(f"Average rank difference: {accuracy['avg_rank_difference']:.2f}")
    print(f"Exact match: {accuracy['exact_match']}")
    print()


def _maybe_plot(
    out_dir: Path,
    per_query_rows: List[Dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["query"] for row in per_query_rows]
    cold_total = [row["cold_total_ms"] for row in per_query_rows]
    cached_total = [row["cached_total_ms"] for row in per_query_rows]

    # Plot 1: cold vs cached totals
    plt.figure(figsize=(10, 4))
    x = list(range(len(labels)))
    width = 0.4
    plt.bar([i - width / 2 for i in x], cold_total, width=width, label="Cold total (ms)")
    plt.bar([i + width / 2 for i in x], cached_total, width=width, label="Cached total (ms)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("ms")
    plt.title("Cold vs Cached Total Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cold_vs_cached_total_latency.png", dpi=160)
    plt.close()

    # Plot 2: breakdown (cold and cached)
    components = ["embedding_ms", "cache_lookup_ms", "vector_db_ms", "reranking_ms"]
    for mode in ("cold", "cached"):
        plt.figure(figsize=(10, 4))
        bottom = [0.0] * len(labels)
        for comp in components:
            values = [row[f"{mode}_{comp}"] for row in per_query_rows]
            plt.bar(labels, values, bottom=bottom, label=comp.replace("_ms", ""))
            bottom = [b + v for b, v in zip(bottom, values)]
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("ms")
        plt.title(f"{mode.capitalize()} Latency Breakdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{mode}_latency_breakdown.png", dpi=160)
        plt.close()


async def run_benchmark(queries: List[str], top_n: int = 5) -> int:
    per_query_rows: List[Dict[str, Any]] = []

    for query in queries:
        # STEP 1 — Cold query
        cache.clear()
        cold = await _run_query(query, top_n=top_n)

        # STEP 2 — Cached query (same query, cache populated)
        cached = await _run_query(query, top_n=top_n)

        accuracy = compare_rankings(cold.doc_ids, cached.doc_ids)
        _print_query_report(query, cold, cached, accuracy)

        cold_lat = cold.response.latency_breakdown
        cached_lat = cached.response.latency_breakdown

        per_query_rows.append(
            {
                "query": query,
                "cold_total_ms": float(cold_lat.total_ms),
                "cached_total_ms": float(cached_lat.total_ms),
                "speedup_pct": float(_latency_improvement_percent(cold_lat.total_ms, cached_lat.total_ms)),
                "top_k_overlap": float(accuracy["top_k_overlap"]),
                "exact_match": bool(accuracy["exact_match"]),
                "cold_embedding_ms": float(cold_lat.embedding_ms),
                "cold_cache_lookup_ms": float(cold_lat.cache_lookup_ms),
                "cold_vector_db_ms": float(cold_lat.vector_db_ms),
                "cold_reranking_ms": float(cold_lat.reranking_ms),
                "cached_embedding_ms": float(cached_lat.embedding_ms),
                "cached_cache_lookup_ms": float(cached_lat.cache_lookup_ms),
                "cached_vector_db_ms": float(cached_lat.vector_db_ms),
                "cached_reranking_ms": float(cached_lat.reranking_ms),
            }
        )

    # AGGREGATE METRICS
    avg_cold = sum(r["cold_total_ms"] for r in per_query_rows) / len(per_query_rows)
    avg_cached = sum(r["cached_total_ms"] for r in per_query_rows) / len(per_query_rows)
    avg_speedup = sum(r["speedup_pct"] for r in per_query_rows) / len(per_query_rows)
    avg_overlap = sum(r["top_k_overlap"] for r in per_query_rows) / len(per_query_rows)
    exact_match_rate = sum(1 for r in per_query_rows if r["exact_match"]) / len(per_query_rows)

    print("-" * 34)
    print("AGGREGATE METRICS")
    print("-" * 34)
    print(f"Average cold latency: {_fmt_ms(avg_cold)}")
    print(f"Average cached latency: {_fmt_ms(avg_cached)}")
    print(f"Average speedup: {_fmt_pct(avg_speedup)}")
    print(f"Average top-k overlap: {avg_overlap * 100:.0f}%")
    print(f"Exact match rate: {exact_match_rate * 100:.0f}%")
    print("-" * 34)

    _maybe_plot(Path("benchmark_results"), per_query_rows)
    return 0


def main() -> int:
    return asyncio.run(run_benchmark(example_queries, top_n=5))


if __name__ == "__main__":
    raise SystemExit(main())

