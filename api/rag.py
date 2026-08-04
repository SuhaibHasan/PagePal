from __future__ import annotations

import asyncio
import hashlib
import json

import anthropic
import redis

from api.config import Settings
from api.schemas import ChatResponse, SourceRef
from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult
from retrieval.reranker import ReciprocalRankFusionReranker

SYSTEM_PROMPT = (
    "You are ProdSupportBuddy, an assistant that helps engineers resolve "
    "production incidents. Answer using only the provided context. If the "
    "context does not contain the answer, say so explicitly and suggest a "
    "next diagnostic step."
)


class RagPipeline:
    def __init__(
        self,
        retrievers: list[BaseRetriever],
        reranker: ReciprocalRankFusionReranker,
        redis_client: redis.Redis,
        settings: Settings,
    ) -> None:
        self._retrievers = retrievers
        self._reranker = reranker
        self._redis = redis_client
        self._settings = settings
        self._llm = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    async def answer(self, query: str) -> ChatResponse:
        cache_key = f"psb:answer:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = self._redis.get(cache_key)
        if cached:
            return ChatResponse.model_validate(json.loads(cached))

        result_lists = await asyncio.gather(
            *(
                asyncio.to_thread(retriever.retrieve, query, self._settings.retrieval_top_k)
                for retriever in self._retrievers
            )
        )
        fused = self._reranker.rerank(list(result_lists), top_k=self._settings.rerank_top_k)

        answer_text = await asyncio.to_thread(self._generate_answer, query, fused)
        response = ChatResponse(
            answer=answer_text,
            sources=[
                SourceRef(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content_snippet=result.content[:280],
                    score=result.score,
                    source_type=result.source_type,
                )
                for result in fused
            ],
        )

        self._redis.setex(cache_key, self._settings.cache_ttl_seconds, response.model_dump_json())
        return response

    def _generate_answer(self, query: str, context: list[RetrievalResult]) -> str:
        if not context:
            context_block = "No relevant context was found in the knowledge base."
        else:
            context_block = "\n\n".join(
                f"[Source {i + 1} | {result.source_type}] {result.content}"
                for i, result in enumerate(context)
            )

        message = self._llm.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"}
            ],
        )
        return "".join(block.text for block in message.content if block.type == "text")
