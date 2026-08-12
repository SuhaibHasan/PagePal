import json

import pytest

import api.rag as rag_module
from api.config import Settings
from api.rag import RagPipeline
from retrieval.answer_generator import Citation
from retrieval.models import RetrievalResult


@pytest.fixture(autouse=True)
def _no_wiki_entry(monkeypatch):
    # These tests exercise the RAG fallback path specifically; the wiki fast path
    # and its background feedback loop are covered separately in tests/api/test_chat_wiki.py.
    async def fake_wiki_retrieve(query):
        return None

    async def fake_wiki_upsert(query, answer, citations):
        pass

    monkeypatch.setattr(rag_module, "wiki_retrieve", fake_wiki_retrieve)
    monkeypatch.setattr(rag_module, "wiki_upsert_from_rag", fake_wiki_upsert)


class _FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def retrieve(self, query, top_k, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self._results


class _FakeReranker:
    def __init__(self, fused: list[RetrievalResult]) -> None:
        self._fused = fused
        self.calls: list[dict] = []

    def rerank(self, query, result_lists, top_k):
        self.calls.append({"query": query, "result_lists": result_lists, "top_k": top_k})
        return self._fused


class _FakeAnswerGenerator:
    def __init__(self, chunks: list[str], citations: list[Citation]) -> None:
        self._chunks = chunks
        self._citations = citations
        self.astream_calls: list[dict] = []

    def top_context(self, context):
        return context[:8]

    async def astream_answer(self, query, context, history=None):
        self.astream_calls.append({"query": query, "context": context, "history": history})
        for chunk in self._chunks:
            yield chunk

    def build_citations(self, answer_text, top_context):
        return self._citations


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.last_ttl = ttl


def make_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, document_id=chunk_id, content="text", score=1.0, source_type="vector"
    )


def make_pipeline(chunks=("Restart ", "the pods."), citations=None, fused=None, retriever_results=None):
    fused = fused if fused is not None else [make_result("a::0")]
    retrievers = {
        "vector": _FakeRetriever(retriever_results or [make_result("a::0")]),
        "keyword": _FakeRetriever(retriever_results or [make_result("a::0")]),
        "graph": _FakeRetriever(retriever_results or []),
    }
    reranker = _FakeReranker(fused)
    answer_generator = _FakeAnswerGenerator(list(chunks), citations or [])
    redis_client = _FakeRedis()
    pipeline = RagPipeline(
        retrievers=retrievers,
        reranker=reranker,
        answer_generator=answer_generator,
        redis_client=redis_client,
        settings=Settings(),
    )
    return pipeline, retrievers, reranker, answer_generator, redis_client


async def _collect(pipeline, query, session_id, filters=None):
    return [event async for event in pipeline.answer_stream(query, session_id, filters)]


async def test_answer_stream_yields_tokens_then_citations_then_done():
    citation = Citation(title="Runbook", url="https://x/a", retrieval_path="vector")
    pipeline, *_ = make_pipeline(chunks=["Restart ", "the pods."], citations=[citation])

    events = await _collect(pipeline, "why is it down", "sess-1")

    assert events[0] == ("token", {"text": "Restart "})
    assert events[1] == ("token", {"text": "the pods."})
    assert events[2] == ("citations", {"citations": [citation.model_dump()]})
    assert events[3] == ("done", {})


async def test_answer_stream_calls_every_retriever_with_filters():
    pipeline, retrievers, *_ = make_pipeline()
    filters = {"service": "payment-service"}

    await _collect(pipeline, "query", "sess-1", filters=filters)

    for retriever in retrievers.values():
        assert retriever.calls[0]["filters"] == filters
        assert retriever.calls[0]["query"] == "query"


async def test_answer_stream_passes_prior_history_to_answer_generator():
    pipeline, _, _, answer_generator, redis_client = make_pipeline()
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "earlier answer"}]
    redis_client.store[RagPipeline._session_key("sess-1")] = json.dumps(history)

    await _collect(pipeline, "follow up", "sess-1")

    assert answer_generator.astream_calls[0]["history"] == history


async def test_answer_stream_appends_new_turn_and_sets_one_hour_ttl():
    pipeline, *_, redis_client = make_pipeline(chunks=["full answer"])

    await _collect(pipeline, "what happened", "sess-1")

    stored = json.loads(redis_client.store[RagPipeline._session_key("sess-1")])
    assert stored == [
        {"role": "user", "content": "what happened"},
        {"role": "assistant", "content": "full answer"},
    ]
    assert redis_client.last_ttl == 3600


async def test_answer_stream_trims_history_to_last_five_turns():
    pipeline, *_, redis_client = make_pipeline(chunks=["new answer"])
    old_history = []
    for i in range(6):
        old_history.append({"role": "user", "content": f"q{i}"})
        old_history.append({"role": "assistant", "content": f"a{i}"})
    redis_client.store[RagPipeline._session_key("sess-1")] = json.dumps(old_history)

    await _collect(pipeline, "latest question", "sess-1")

    stored = json.loads(redis_client.store[RagPipeline._session_key("sess-1")])
    assert len(stored) == 10
    assert stored[-2:] == [
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "new answer"},
    ]
    # oldest turn (q0/a0) must have been evicted to make room
    assert {"role": "user", "content": "q0"} not in stored


# --- Resilience: individual dependency failures must not fail the whole request ---


class _FailingRetriever:
    def retrieve(self, query, top_k, filters=None):
        raise ConnectionError("neo4j unreachable")


async def test_one_failing_retriever_does_not_fail_the_whole_answer():
    pipeline, retrievers, *_ = make_pipeline(chunks=["answer despite one dead source"])
    # RagPipeline holds the same dict object make_pipeline built, so mutating
    # it here is reflected in the pipeline too.
    retrievers["graph"] = _FailingRetriever()

    events = await _collect(pipeline, "why is it down", "sess-1")

    tokens = "".join(payload["text"] for event, payload in events if event == "token")
    assert tokens == "answer despite one dead source"
    assert events[-1] == ("done", {})


async def test_wiki_lookup_failure_falls_back_to_full_rag(monkeypatch):
    async def failing_wiki_retrieve(query):
        raise ConnectionError("elasticsearch unreachable")

    monkeypatch.setattr(rag_module, "wiki_retrieve", failing_wiki_retrieve)
    pipeline, retrievers, *_ = make_pipeline(chunks=["RAG answer, wiki was unreachable"])

    events = await _collect(pipeline, "why is it down", "sess-1")

    tokens = "".join(payload["text"] for event, payload in events if event == "token")
    assert tokens == "RAG answer, wiki was unreachable"
    # proves the RAG path actually ran, not just that nothing crashed
    assert retrievers["vector"].calls


class _DownRedis:
    def get(self, key):
        raise ConnectionError("redis unreachable")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis unreachable")


async def test_redis_outage_during_history_load_degrades_to_no_history():
    pipeline, _, _, answer_generator, _ = make_pipeline(chunks=["answer without history"])
    pipeline._redis = _DownRedis()

    events = await _collect(pipeline, "follow up", "sess-1")

    assert answer_generator.astream_calls[0]["history"] == []
    assert events[-1] == ("done", {})


async def test_redis_outage_during_history_save_does_not_fail_the_request():
    pipeline, *_ = make_pipeline(chunks=["answer that can't be persisted"])
    pipeline._redis = _DownRedis()

    events = await _collect(pipeline, "what happened", "sess-1")

    tokens = "".join(payload["text"] for event, payload in events if event == "token")
    assert tokens == "answer that can't be persisted"
    assert events[-1] == ("done", {})
