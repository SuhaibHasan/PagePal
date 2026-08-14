from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import redis

from api.config import Settings
from api.telemetry import tracer
from retrieval.answer_generator import AnswerGenerator, HistoryTurn
from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult
from retrieval.reranker import Reranker
from wiki.retriever import retrieve as wiki_retrieve
from wiki.schema import WikiEntry
from wiki.writer import upsert_from_rag as wiki_upsert_from_rag

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 3600
MAX_HISTORY_TURNS = 5


class RagPipeline:
    def __init__(
        self,
        retrievers: dict[str, BaseRetriever],
        reranker: Reranker,
        answer_generator: AnswerGenerator,
        redis_client: redis.Redis,
        settings: Settings,
    ) -> None:
        self._retrievers = retrievers
        self._reranker = reranker
        self._answer_generator = answer_generator
        self._redis = redis_client
        self._settings = settings
        # Holds references to fire-and-forget wiki-feedback tasks so they aren't
        # garbage-collected before they run.
        self._background_tasks: set[asyncio.Task] = set()

    async def answer_stream(
        self, query: str, session_id: str, filters: dict[str, str] | None = None
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yields (event, payload) pairs: zero or more "token", then "citations", then "done"."""
        with tracer.start_as_current_span("query_routing") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("query", query)

            history = await self._load_history(session_id)

            # 1. Try the wiki first - a pre-validated, high-confidence entry skips
            # the full retrieval + rerank + generation path entirely. A lookup
            # failure (e.g. Elasticsearch unreachable) degrades to "no wiki hit"
            # rather than failing the whole chat request.
            try:
                wiki_entry = await wiki_retrieve(query)
            except Exception:
                logger.warning("Wiki lookup failed; falling back to full RAG", exc_info=True)
                wiki_entry = None

            if wiki_entry is not None:
                async for event, payload in self._answer_from_wiki(
                    query, session_id, wiki_entry, history
                ):
                    yield event, payload
                return

            # 2. Existing RAG logic, unchanged.
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

            citation_dicts = [citation.model_dump() for citation in citations]

            # 3. Feed the RAG answer back into the wiki, in the background.
            self._queue_wiki_feedback(query, answer_text, citation_dicts)

            yield "citations", {"citations": citation_dicts}
            yield "done", {}

    async def _answer_from_wiki(
        self, query: str, session_id: str, wiki_entry: WikiEntry, history: list[HistoryTurn]
    ) -> AsyncIterator[tuple[str, dict]]:
        answer_chunks: list[str] = []
        with tracer.start_as_current_span("llm_call"):
            async for chunk in self._answer_generator.from_wiki(query, wiki_entry, history=history):
                answer_chunks.append(chunk)
                yield "token", {"text": chunk}

        answer_text = "".join(answer_chunks)
        citations = self._answer_generator.build_wiki_citations(wiki_entry)

        await self._append_turn(session_id, query, answer_text)

        yield "citations", {"citations": [citation.model_dump() for citation in citations]}
        yield "done", {}

    def _queue_wiki_feedback(self, query: str, answer_text: str, citations: list[dict]) -> None:
        task = asyncio.create_task(wiki_upsert_from_rag(query, answer_text, citations))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _retrieve(
        self, name: str, retriever: BaseRetriever, query: str, filters: dict[str, str] | None
    ) -> list[RetrievalResult]:
        with tracer.start_as_current_span(f"retrieval.{name}"):
            try:
                return await asyncio.to_thread(
                    retriever.retrieve, query, self._settings.retrieval_top_k, filters
                )
            except Exception:
                # One source being down (e.g. Neo4j) shouldn't take the other
                # two with it - asyncio.gather would otherwise fail the whole
                # batch on the first exception.
                logger.warning(
                    "Retrieval source %r failed; continuing without it", name, exc_info=True
                )
                return []

    async def _load_history(self, session_id: str) -> list[HistoryTurn]:
        try:
            raw = await asyncio.to_thread(self._redis.get, self._session_key(session_id))
        except Exception:
            logger.warning("Redis unavailable; continuing without session history", exc_info=True)
            return []
        if not raw:
            return []
        return json.loads(raw)

    async def _append_turn(self, session_id: str, question: str, answer: str) -> None:
        history = await self._load_history(session_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        trimmed = history[-(MAX_HISTORY_TURNS * 2) :]
        try:
            await asyncio.to_thread(
                self._redis.setex,
                self._session_key(session_id),
                SESSION_TTL_SECONDS,
                json.dumps(trimmed),
            )
        except Exception:
            logger.warning(
                "Redis unavailable; could not persist session history", exc_info=True
            )

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"psb:session:{session_id}"
