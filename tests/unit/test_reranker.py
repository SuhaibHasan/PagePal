from retrieval.models import RetrievalResult
from retrieval.reranker import ReciprocalRankFusionReranker


def make_result(chunk_id: str, source_type: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        content=f"content for {chunk_id}",
        score=score,
        source_type=source_type,
    )


def test_rerank_boosts_chunks_present_in_multiple_lists():
    vector_results = [make_result("a", "vector", 0.9), make_result("b", "vector", 0.8)]
    keyword_results = [make_result("b", "keyword", 12.0), make_result("c", "keyword", 10.0)]

    reranker = ReciprocalRankFusionReranker(k=60)
    fused = reranker.rerank([vector_results, keyword_results], top_k=3)

    assert fused[0].chunk_id == "b"
    assert {result.chunk_id for result in fused} == {"a", "b", "c"}


def test_rerank_respects_top_k():
    results = [make_result(str(i), "vector", float(i)) for i in range(10)]

    reranker = ReciprocalRankFusionReranker()
    fused = reranker.rerank([results], top_k=3)

    assert len(fused) == 3


def test_rerank_empty_lists_returns_empty():
    reranker = ReciprocalRankFusionReranker()

    assert reranker.rerank([], top_k=5) == []
    assert reranker.rerank([[], []], top_k=5) == []
