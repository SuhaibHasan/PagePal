from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import redis

from api.config import Settings
from api.telemetry import tracer
from retrieval.answer_generator import AnswerGenerator, HistoryTurn
from retrieval.base import BaseRetriever
from retrieval.reranker import CrossEncoderReranker

SESSION_TTL_SECONDS = 3600
MAX_HISTORY_TURNS = 5


class RagPipeline:
    def __init__(
        self,
        retrievers: dict[str, BaseRetriever],
        reranker: CrossEncoderReranker,
        answer_generator: AnswerGenerator,
        redis_client: redis.Redis,
        settings: Settings,
    ) -> None:
        self._retrievers = retrievers
        self._reranker = reranker
        self._answer_generator = answer_generator
        self._redis = redis_client
        self._settings = settings

    async def answer_stream(
        self, query: str, session_id: str, filters: dict[str, str] | None = None
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yields (event, payload) pairs: zero or more "token", then "citations", then "done"."""
        with tracer.start_as_current_span("query_routing") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("query", query)

            history = await self._load_history(session_id)

            result_lists = await asyncio.gather(
                *(
                    self._retrieve(name, retriever, query, filters)
                    for name, retriever in self._retrievers.items()
                )
            )

            with tracer.start_as_current_span("reranking"):
                fused = await asyncio.to_thread(
                    self._reranker.rerank, query, list(result_lists), self._settings.rerank_top_k
                )

            top_context = self._answer_generator.top_context(fused)

            answer_chunks: list[str] = []
            with tracer.start_as_current_span("llm_call"):
                async for chunk in self._answer_generator.astream_answer(
                    query, fused, history=history
                ):
                    answer_chunks.append(chunk)
                    yield "token", {"text": chunk}

            answer_text = "".join(answer_chunks)
            citations = self._answer_generator.build_citations(answer_text, top_context)

            await self._append_turn(session_id, query, answer_text)

            yield "citations", {"citations": [citation.model_dump() for citation in citations]}
            yield "done", {}

    async def _retrieve(
        self, name: str, retriever: BaseRetriever, query: str, filters: dict[str, str] | None
    ):
        with tracer.start_as_current_span(f"retrieval.{name}"):
            return await asyncio.to_thread(
                retriever.retrieve, query, self._settings.retrieval_top_k, filters
            )

    async def _load_history(self, session_id: str) -> list[HistoryTurn]:
        raw = await asyncio.to_thread(self._redis.get, self._session_key(session_id))
        if not raw:
            return []
        return json.loads(raw)

    async def _append_turn(self, session_id: str, question: str, answer: str) -> None:
        history = await self._load_history(session_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        trimmed = history[-(MAX_HISTORY_TURNS * 2) :]
        await asyncio.to_thread(
            self._redis.setex,
            self._session_key(session_id),
            SESSION_TTL_SECONDS,
            json.dumps(trimmed),
        )

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"psb:session:{session_id}"
