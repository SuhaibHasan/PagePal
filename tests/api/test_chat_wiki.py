from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.rag as rag_module
from api.config import get_settings
from api.dependencies import get_redis_client
from api.main import app, get_rag_pipeline
from api.rag import RagPipeline
from retrieval.answer_generator import Citation
from wiki.schema import WikiEntry


class _FakeRedis:
    def __init__(self) -> None:
        self._sets: dict[str, dict[str, float]] = {}
        self._values: dict[str, str] = {}

    def zremrangebyscore(self, key, lo, hi):
        members = self._sets.get(key, {})
        self._sets[key] = {m: s for m, s in members.items() if not (lo <= s <= hi)}

    def zcard(self, key):
        return len(self._sets.get(key, {}))

    def zadd(self, key, mapping):
        self._sets.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        pass

    def get(self, key):
        return self._values.get(key)

    def setex(self, key, ttl, value):
        self._values[key] = value


class _SpyRetriever:
    def __init__(self, results: list | None = None) -> None:
        self.calls: list[dict] = []
        self._results = results or []

    def retrieve(self, query, top_k=10, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self._results


class _ForbiddenReranker:
    def rerank(self, query, result_lists, top_k):
        raise AssertionError("reranker must not be called on the wiki fast path")


class _FakeAnswerGenerator:
    def __init__(self, wiki_chunks: list[str]) -> None:
        self._wiki_chunks = wiki_chunks
        self.from_wiki_calls: list[dict] = []

    async def from_wiki(self, query, entry, history=None):
        self.from_wiki_calls.append({"query": query, "entry": entry, "history": history})
        for chunk in self._wiki_chunks:
            yield chunk

    def build_wiki_citations(self, entry) -> list[Citation]:
        return [Citation(title=entry.title, retrieval_path="wiki")]

    def top_context(self, context):
        raise AssertionError("top_context must not be called on the wiki fast path")

    async def astream_answer(self, query, context, history=None):
        raise AssertionError("astream_answer must not be called on the wiki fast path")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    def build_citations(self, answer_text, top_context):
        raise AssertionError("build_citations must not be called on the wiki fast path")


def make_wiki_entry(**overrides) -> WikiEntry:
    now = datetime.now(UTC)
    defaults = {
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "resolution_steps": ["Restart payment-service pods"],
        "confidence": 0.9,
        "created_at": now,
        "last_updated": now,
        "last_validated": now,
    }
    defaults.update(overrides)
    return WikiEntry(**defaults)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_wiki_fast_path_skips_retrievers_and_tags_citation_as_wiki(monkeypatch, client):
    entry = make_wiki_entry()

    async def fake_wiki_retrieve(query: str):
        return entry

    monkeypatch.setattr(rag_module, "wiki_retrieve", fake_wiki_retrieve)

    vector_retriever = _SpyRetriever()
    keyword_retriever = _SpyRetriever()
    graph_retriever = _SpyRetriever()
    answer_generator = _FakeAnswerGenerator(["Restart ", "the pods."])
    fake_redis = _FakeRedis()

    pipeline = RagPipeline(
        retrievers={
            "vector": vector_retriever,
            "keyword": keyword_retriever,
            "graph": graph_retriever,
        },
        reranker=_ForbiddenReranker(),
        answer_generator=answer_generator,
        redis_client=fake_redis,
        settings=get_settings(),
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    # The /chat route also depends on get_redis_client directly, for rate limiting.
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    response = client.post(
        "/chat", json={"message": "why are payments failing", "session_id": "sess-1"}
    )

    assert response.status_code == 200
    body = response.text
    assert "event: token\ndata: {\"text\": \"Restart \"}" in body
    assert "event: done\ndata: {}" in body

    assert vector_retriever.calls == []
    assert keyword_retriever.calls == []
    assert graph_retriever.calls == []

    assert '"retrieval_path": "wiki"' in body
    assert answer_generator.from_wiki_calls[0]["query"] == "why are payments failing"
    assert answer_generator.from_wiki_calls[0]["entry"] is entry


def test_falls_back_to_normal_rag_when_no_wiki_entry_found(monkeypatch, client):
    async def fake_wiki_retrieve(query: str):
        return None

    monkeypatch.setattr(rag_module, "wiki_retrieve", fake_wiki_retrieve)

    async def fake_wiki_upsert(query, answer, citations):
        pass

    monkeypatch.setattr(rag_module, "wiki_upsert_from_rag", fake_wiki_upsert)

    vector_retriever = _SpyRetriever()
    keyword_retriever = _SpyRetriever()
    graph_retriever = _SpyRetriever()

    class _NoOpReranker:
        def rerank(self, query, result_lists, top_k):
            return []

    class _EmptyAnswerGenerator:
        def top_context(self, context):
            return context

        async def astream_answer(self, query, context, history=None):
            yield "no wiki entry, using RAG"

        def build_citations(self, answer_text, top_context):
            return []

        async def from_wiki(self, query, entry, history=None):
            raise AssertionError("from_wiki must not be called when no wiki entry was found")
            yield  # pragma: no cover

        def build_wiki_citations(self, entry):
            raise AssertionError("build_wiki_citations must not be called without a wiki entry")

    fake_redis = _FakeRedis()
    pipeline = RagPipeline(
        retrievers={
            "vector": vector_retriever,
            "keyword": keyword_retriever,
            "graph": graph_retriever,
        },
        reranker=_NoOpReranker(),
        answer_generator=_EmptyAnswerGenerator(),
        redis_client=fake_redis,
        settings=get_settings(),
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    response = client.post("/chat", json={"message": "hello", "session_id": "sess-2"})

    assert response.status_code == 200
    assert vector_retriever.calls[0]["query"] == "hello"
    assert keyword_retriever.calls[0]["query"] == "hello"
    assert graph_retriever.calls[0]["query"] == "hello"
    assert "no wiki entry, using RAG" in response.text
