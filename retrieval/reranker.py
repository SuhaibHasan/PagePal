from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sentence_transformers import CrossEncoder

from retrieval.models import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self, query: str, result_lists: list[list[RetrievalResult]], top_k: int = 8
    ) -> list[RetrievalResult]: ...

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Each retriever is expected to cap itself around this already; this is a
# defensive limit so a misbehaving retriever can't blow up cross-encoder cost.
MAX_CANDIDATES_PER_RETRIEVER = 10


@dataclass
class _Candidate:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict
    source_types: set[str] = field(default_factory=set)


class CrossEncoderReranker:
    def __init__(self, model: CrossEncoder | None = None, model_name: str = CROSS_ENCODER_MODEL) -> None:
        # model is injectable so tests can supply a fake .predict() without
        # downloading the real model - inference itself is fully local either
        # way, no external API call.
        self._model = model or CrossEncoder(model_name)

    def rerank(
        self, query: str, result_lists: list[list[RetrievalResult]], top_k: int = 8
    ) -> list[RetrievalResult]:
        candidates = self._deduplicate(result_lists)
        if not candidates:
            return []

        pairs = [(query, candidate.content) for candidate in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                score=float(score),
                source_type="+".join(sorted(candidate.source_types)),
                metadata=candidate.metadata,
            )
            for candidate, score in ranked[:top_k]
        ]

    @staticmethod
    def _deduplicate(result_lists: list[list[RetrievalResult]]) -> list[_Candidate]:
        candidates: dict[str, _Candidate] = {}
        for results in result_lists:
            for result in results[:MAX_CANDIDATES_PER_RETRIEVER]:
                existing = candidates.get(result.chunk_id)
                if existing is None:
                    candidates[result.chunk_id] = _Candidate(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        content=result.content,
                        metadata=dict(result.metadata),
                        source_types={result.source_type},
                    )
                    continue

                existing.source_types.add(result.source_type)
                existing.metadata.update(result.metadata)
                # Prefer the richer content - e.g. the graph retriever's content
                # is its causal path prefixed onto the same underlying chunk text.
                if len(result.content) > len(existing.content):
                    existing.content = result.content

        return list(candidates.values())


class PassthroughReranker:
    """Used when the cross-encoder can't be loaded (e.g. no network access to
    HuggingFace Hub). Dedupes the same way CrossEncoderReranker does, but keeps
    each candidate's first-seen retrieval order instead of a learned score -
    degraded ranking quality beats no answer at all."""

    def rerank(
        self, query: str, result_lists: list[list[RetrievalResult]], top_k: int = 8
    ) -> list[RetrievalResult]:
        candidates = CrossEncoderReranker._deduplicate(result_lists)
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                score=0.0,
                source_type="+".join(sorted(candidate.source_types)),
                metadata=candidate.metadata,
            )
            for candidate in candidates[:top_k]
        ]
