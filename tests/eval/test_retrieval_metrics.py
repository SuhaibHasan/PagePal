import pytest

from retrieval.models import RetrievalResult
from tests.eval.metrics import hit_rate_at_k, mean_reciprocal_rank


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self._results_by_query = results_by_query

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self._results_by_query.get(query, [])[:top_k]


def make_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, document_id=chunk_id, content="x", score=1.0, source_type="vector"
    )


def test_hit_rate_at_k_counts_expected_chunk_presence():
    retriever = FakeRetriever(
        {
            "q1": [make_result("a"), make_result("expected-1")],
            "q2": [make_result("b")],
        }
    )
    queries = {"q1": "expected-1", "q2": "expected-2"}

    assert hit_rate_at_k(retriever, queries, top_k=5) == 0.5


def test_mrr_rewards_higher_rank():
    retriever = FakeRetriever(
        {
            "q1": [make_result("expected-1")],
            "q2": [make_result("a"), make_result("expected-2")],
        }
    )
    queries = {"q1": "expected-1", "q2": "expected-2"}

    assert mean_reciprocal_rank(retriever, queries) == pytest.approx(0.75)


def test_metrics_on_empty_queries_return_zero():
    retriever = FakeRetriever({})

    assert hit_rate_at_k(retriever, {}) == 0.0
    assert mean_reciprocal_rank(retriever, {}) == 0.0
