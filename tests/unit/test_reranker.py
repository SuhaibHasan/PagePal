from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker


class _FakeCrossEncoder:
    def __init__(self, scores_by_content: dict[str, float] | None = None, default_score: float = 0.0) -> None:
        self._scores_by_content = scores_by_content or {}
        self._default_score = default_score
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        pairs = list(pairs)
        self.predict_calls.append(pairs)
        return [self._scores_by_content.get(content, self._default_score) for _, content in pairs]


def make_result(chunk_id: str, source_type: str, content: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        content=content if content is not None else f"content for {chunk_id}",
        score=0.0,
        source_type=source_type,
        metadata={f"{source_type}_meta": True},
    )


def test_rerank_deduplicates_by_chunk_id_and_combines_source_tags():
    vector_results = [make_result("a", "vector"), make_result("b", "vector")]
    keyword_results = [make_result("b", "keyword"), make_result("c", "keyword")]
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    fused = reranker.rerank("query", [vector_results, keyword_results])

    by_id = {result.chunk_id: result for result in fused}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].source_type == "vector"
    assert by_id["b"].source_type == "keyword+vector"
    assert by_id["c"].source_type == "keyword"


def test_rerank_merges_metadata_from_all_contributing_retrievers():
    vector_results = [make_result("a", "vector")]
    keyword_results = [make_result("a", "keyword")]
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    fused = reranker.rerank("query", [vector_results, keyword_results])

    assert fused[0].metadata == {"vector_meta": True, "keyword_meta": True}


def test_rerank_orders_results_by_cross_encoder_score():
    results = [
        make_result("a", "vector", content="low relevance"),
        make_result("b", "vector", content="high relevance"),
        make_result("c", "vector", content="medium relevance"),
    ]
    model = _FakeCrossEncoder(
        {"low relevance": 0.1, "high relevance": 9.5, "medium relevance": 4.0}
    )
    reranker = CrossEncoderReranker(model=model)

    fused = reranker.rerank("query", [results])

    assert [result.chunk_id for result in fused] == ["b", "c", "a"]
    assert fused[0].score == 9.5


def test_rerank_defaults_to_top_8():
    results = [make_result(str(i), "vector") for i in range(12)]
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    fused = reranker.rerank("query", [results])

    assert len(fused) == 8


def test_rerank_respects_explicit_top_k():
    results = [make_result(str(i), "vector") for i in range(12)]
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    fused = reranker.rerank("query", [results], top_k=3)

    assert len(fused) == 3


def test_rerank_truncates_each_retriever_list_to_ten_candidates():
    results = [make_result(str(i), "vector") for i in range(15)]
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("query", [results], top_k=20)

    assert len(model.predict_calls[0]) == 10


def test_rerank_prefers_richer_content_when_deduplicating():
    short_result = make_result("a", "vector", content="chunk text")
    path_prefixed_result = make_result(
        "a", "graph", content="auth-service → DEPENDS_ON → redis-cache\n\nchunk text"
    )
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    fused = reranker.rerank("query", [[short_result], [path_prefixed_result]])

    assert fused[0].content == "auth-service → DEPENDS_ON → redis-cache\n\nchunk text"
    assert fused[0].source_type == "graph+vector"


def test_rerank_pairs_query_with_content_for_the_model():
    results = [make_result("a", "vector", content="chunk text")]
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("why is payment-service failing", [results])

    assert model.predict_calls[0] == [("why is payment-service failing", "chunk text")]


def test_rerank_empty_lists_returns_empty():
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder())

    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [[], []]) == []
