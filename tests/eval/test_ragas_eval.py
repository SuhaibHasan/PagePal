import math
from types import SimpleNamespace

import pytest

pytest.importorskip("ragas", reason="requires `uv sync --group eval`")

from retrieval.models import RetrievalResult
from tests.eval.golden_dataset import GoldenQA
from tests.eval.ragas_eval import (
    METRIC_NAMES,
    build_samples,
    mean_scores,
    render_category_table,
    render_comparison_table,
    scores_by_category,
)


class _FakeResult:
    def __init__(self, scores: list[dict[str, float]]) -> None:
        self.scores = scores

    def __getitem__(self, key: str) -> list[float]:
        return [row[key] for row in self.scores]


def make_result(rows: list[dict[str, float]]) -> _FakeResult:
    return _FakeResult(rows)


def test_mean_scores_averages_each_metric_across_samples():
    result = make_result(
        [
            {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_precision": 0.8, "context_recall": 0.9},
            {"faithfulness": 0.0, "answer_relevancy": 0.5, "context_precision": 0.6, "context_recall": 0.7},
        ]
    )

    scores = mean_scores(result)

    assert scores["faithfulness"] == 0.5
    assert scores["answer_relevancy"] == 0.5
    assert scores["context_precision"] == 0.7
    assert scores["context_recall"] == 0.8


def test_mean_scores_ignores_nan_values():
    result = make_result(
        [
            {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0},
            {"faithfulness": float("nan"), "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0},
        ]
    )

    scores = mean_scores(result)

    assert scores["faithfulness"] == 1.0


def test_mean_scores_returns_nan_when_every_value_is_nan():
    result = make_result([{name: float("nan") for name in METRIC_NAMES}])

    scores = mean_scores(result)

    assert math.isnan(scores["faithfulness"])


def test_scores_by_category_groups_by_the_golden_qas_category():
    golden = [
        GoldenQA(id="a", category="semantic", question="q1", ground_truth="g1"),
        GoldenQA(id="b", category="exact_match", question="q2", ground_truth="g2"),
        GoldenQA(id="c", category="semantic", question="q3", ground_truth="g3"),
    ]
    result = make_result(
        [
            {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0},
            {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0},
            {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5},
        ]
    )

    grouped = scores_by_category(golden, result)

    assert set(grouped.keys()) == {"semantic", "exact_match"}
    assert grouped["semantic"]["faithfulness"] == 0.75  # mean of samples a and c
    assert grouped["exact_match"]["faithfulness"] == 0.0


def test_render_comparison_table_formats_values_and_missing_scores():
    scores_by_config = {
        "Vector-only": {"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7, "context_recall": 0.6},
        "Graph-only": {"faithfulness": float("nan"), "answer_relevancy": 0.5, "context_precision": 0.4, "context_recall": 0.3},
    }

    table = render_comparison_table(scores_by_config)

    assert "| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall |" in table
    assert "| Vector-only | 0.800 | 0.900 | 0.700 | 0.600 |" in table
    assert "| Graph-only | N/A | 0.500 | 0.400 | 0.300 |" in table


def test_render_category_table_includes_every_config_category_pair():
    category_scores = {
        "Vector-only": {
            "semantic": {name: 0.5 for name in METRIC_NAMES},
            "exact_match": {name: 0.25 for name in METRIC_NAMES},
        },
    }

    table = render_category_table(category_scores)

    assert "| Vector-only | Exact Match | 0.250 | 0.250 | 0.250 | 0.250 |" in table
    assert "| Vector-only | Semantic | 0.500 | 0.500 | 0.500 | 0.500 |" in table


def make_retrieval_result(chunk_id: str, content: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, document_id=chunk_id, content=content, score=1.0, source_type="vector"
    )


class _FakeAnswerGenerator:
    def top_context(self, context):
        return context[:2]

    def generate(self, question, context):
        return SimpleNamespace(answer=f"answer to: {question}")


def test_build_samples_uses_retrieved_content_and_generated_answer():
    golden = [GoldenQA(id="a", category="semantic", question="why is it down", ground_truth="ref answer")]

    def retrieve(query: str) -> list[RetrievalResult]:
        return [make_retrieval_result("c1", "relevant chunk text")]

    samples = build_samples(retrieve, _FakeAnswerGenerator(), golden)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.user_input == "why is it down"
    assert sample.retrieved_contexts == ["relevant chunk text"]
    assert sample.response == "answer to: why is it down"
    assert sample.reference == "ref answer"


def test_build_samples_falls_back_to_placeholder_when_nothing_retrieved():
    golden = [GoldenQA(id="a", category="semantic", question="q", ground_truth="g")]

    samples = build_samples(lambda query: [], _FakeAnswerGenerator(), golden)

    assert samples[0].retrieved_contexts == ["(no context retrieved)"]
