from __future__ import annotations

import asyncio
import hashlib
import json

import redis

from api.config import Settings
from api.schemas import ChatResponse
from retrieval.answer_generator import AnswerGenerator
from retrieval.base import BaseRetriever
from retrieval.reranker import CrossEncoderReranker


class RagPipeline:
    def __init__(
        self,
        retrievers: list[BaseRetriever],
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
        fused = await asyncio.to_thread(
            self._reranker.rerank, query, list(result_lists), self._settings.rerank_top_k
        )

        result = await asyncio.to_thread(self._answer_generator.generate, query, fused)
        response = ChatResponse(answer=result.answer, citations=result.citations)

        self._redis.setex(cache_key, self._settings.cache_ttl_seconds, response.model_dump_json())
        return response
