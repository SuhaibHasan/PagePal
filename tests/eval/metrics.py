from __future__ import annotations

from retrieval.base import BaseRetriever


def hit_rate_at_k(retriever: BaseRetriever, queries: dict[str, str], top_k: int = 5) -> float:
    if not queries:
        return 0.0
    hits = 0
    for query, expected_chunk_id in queries.items():
        results = retriever.retrieve(query, top_k=top_k)
        if any(result.chunk_id == expected_chunk_id for result in results):
            hits += 1
    return hits / len(queries)


def mean_reciprocal_rank(retriever: BaseRetriever, queries: dict[str, str], top_k: int = 10) -> float:
    if not queries:
        return 0.0
    reciprocal_ranks = []
    for query, expected_chunk_id in queries.items():
        results = retriever.retrieve(query, top_k=top_k)
        rank = next(
            (i + 1 for i, result in enumerate(results) if result.chunk_id == expected_chunk_id),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
