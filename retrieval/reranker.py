from __future__ import annotations

from collections import defaultdict

from retrieval.models import RetrievalResult


class ReciprocalRankFusionReranker:
    def __init__(self, k: int = 60) -> None:
        self._k = k

    def rerank(
        self, result_lists: list[list[RetrievalResult]], top_k: int = 10
    ) -> list[RetrievalResult]:
        fused_scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                fused_scores[result.chunk_id] += 1.0 / (self._k + rank)
                if result.chunk_id not in best_result:
                    best_result[result.chunk_id] = result

        ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        return [
            best_result[chunk_id].model_copy(update={"score": fused_scores[chunk_id]})
            for chunk_id in ranked_ids[:top_k]
        ]
