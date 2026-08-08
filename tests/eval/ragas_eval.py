"""RAGAS evaluation comparing vector-only, keyword-only, graph-only, and hybrid retrieval.

Usage (requires docker-compose services running, populated, and ANTHROPIC_API_KEY set):

    uv sync --group eval
    uv run python -m ingestion --local-dir tests/eval/fixtures/runbooks
    uv run python tests/eval/ragas_eval.py [--limit N] [--output report.md]

This makes real Claude API calls for every generated answer *and* for every RAGAS metric
judgment (4 metrics x 20 questions x 4 retrieval configs), so a full run takes several
minutes and is not meant to run in CI on every commit - it's a deliberate, manually-triggered
quality report.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from statistics import fmean

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from api.config import get_settings
from api.dependencies import (
    get_answer_generator,
    get_graph_retriever,
    get_keyword_retriever,
    get_reranker,
    get_vector_retriever,
)
from retrieval.answer_generator import AnswerGenerator
from retrieval.models import RetrievalResult
from tests.eval.golden_dataset import GOLDEN_DATASET, GoldenQA

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = [metric.name for metric in METRICS]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RetrieverFn = Callable[[str], list[RetrievalResult]]


def build_retrieval_configs() -> dict[str, RetrieverFn]:
    vector = get_vector_retriever()
    keyword = get_keyword_retriever()
    graph = get_graph_retriever()
    reranker = get_reranker()

    def hybrid(query: str) -> list[RetrievalResult]:
        result_lists = [
            vector.retrieve(query, top_k=10),
            keyword.retrieve(query, top_k=10),
            graph.retrieve(query, top_k=10),
        ]
        return reranker.rerank(query, result_lists, top_k=8)

    return {
        "Vector-only": lambda query: vector.retrieve(query, top_k=8),
        "Keyword-only": lambda query: keyword.retrieve(query, top_k=8),
        "Graph-only": lambda query: graph.retrieve(query, top_k=8),
        "Hybrid": hybrid,
    }


def build_ragas_llm() -> LangchainLLMWrapper:
    settings = get_settings()
    chat = ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    # Same model family the app embeds chunks/queries with (all-MiniLM-L6-v2), just
    # loaded through langchain since answer_relevancy needs a langchain Embeddings object.
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))


def build_samples(
    retrieve: RetrieverFn, answer_generator: AnswerGenerator, golden: list[GoldenQA]
) -> list[SingleTurnSample]:
    samples = []
    for qa in golden:
        context = retrieve(qa.question)
        top_context = answer_generator.top_context(context)
        result = answer_generator.generate(qa.question, context)
        contexts = [item.content for item in top_context] or ["(no context retrieved)"]
        samples.append(
            SingleTurnSample(
                user_input=qa.question,
                retrieved_contexts=contexts,
                response=result.answer,
                reference=qa.ground_truth,
            )
        )
    return samples


def run_ragas(samples: list[SingleTurnSample], llm, embeddings) -> EvaluationResult:
    dataset = EvaluationDataset(samples=samples)
    return evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=embeddings)


def _safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return fmean(clean) if clean else float("nan")


def mean_scores(result: EvaluationResult) -> dict[str, float]:
    return {name: _safe_mean(result[name]) for name in METRIC_NAMES}


def scores_by_category(golden: list[GoldenQA], result: EvaluationResult) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for qa, score_row in zip(golden, result.scores, strict=True):
        grouped[qa.category].append(score_row)

    return {
        category: {name: _safe_mean([row[name] for row in rows]) for name in METRIC_NAMES}
        for category, rows in grouped.items()
    }


def _format_cell(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.3f}"


def render_comparison_table(scores_by_config: dict[str, dict[str, float]]) -> str:
    header = ["Configuration", *[name.replace("_", " ").title() for name in METRIC_NAMES]]
    lines = [
        f"| {' | '.join(header)} |",
        f"|{'---|' * len(header)}",
    ]
    for config_name, scores in scores_by_config.items():
        cells = [config_name, *[_format_cell(scores[name]) for name in METRIC_NAMES]]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def render_category_table(
    category_scores_by_config: dict[str, dict[str, dict[str, float]]],
) -> str:
    header = ["Configuration", "Category", *[name.replace("_", " ").title() for name in METRIC_NAMES]]
    lines = [
        f"| {' | '.join(header)} |",
        f"|{'---|' * len(header)}",
    ]
    for config_name, by_category in category_scores_by_config.items():
        for category, scores in sorted(by_category.items()):
            cells = [
                config_name,
                category.replace("_", " ").title(),
                *[_format_cell(scores[name]) for name in METRIC_NAMES],
            ]
            lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N golden questions"
    )
    parser.add_argument("--output", type=str, default=None, help="Write the markdown report here")
    args = parser.parse_args()

    golden = GOLDEN_DATASET[: args.limit] if args.limit else GOLDEN_DATASET

    llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()
    answer_generator = get_answer_generator()
    configs = build_retrieval_configs()

    overall_scores: dict[str, dict[str, float]] = {}
    category_scores: dict[str, dict[str, dict[str, float]]] = {}

    for config_name, retrieve in configs.items():
        print(f"Running {config_name} ({len(golden)} questions)...", file=sys.stderr)
        samples = build_samples(retrieve, answer_generator, golden)
        result = run_ragas(samples, llm, embeddings)
        overall_scores[config_name] = mean_scores(result)
        category_scores[config_name] = scores_by_category(golden, result)

    counts = {category: sum(1 for qa in golden if qa.category == category) for category in ("semantic", "exact_match", "relationship")}
    summary_line = (
        f"Golden dataset: {len(golden)} questions ({counts['semantic']} semantic, "
        f"{counts['exact_match']} exact-match, {counts['relationship']} relationship)"
    )

    report = [
        "# RAGAS evaluation: retrieval configuration comparison",
        "",
        summary_line,
        "",
        "## Overall",
        "",
        render_comparison_table(overall_scores),
        "",
        "## By query category",
        "",
        render_category_table(category_scores),
    ]
    output = "\n".join(report)

    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nReport written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
