from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.dependencies import (
    get_graph_retriever,
    get_ingestion_pipeline,
    get_redis_client,
)
from api.main import WIKI_INVALIDATION_INTERVAL_HOURS, app, get_rag_pipeline
from retrieval.reranker import PassthroughReranker
from wiki.invalidator import invalidate_stale


class _FakeRedis:
    def __init__(self) -> None:
        self._sets: dict[str, dict[str, float]] = {}

    def zremrangebyscore(self, key, lo, hi):
        members = self._sets.get(key, {})
        self._sets[key] = {m: s for m, s in members.items() if not (lo <= s <= hi)}

    def zcard(self, key):
        return len(self._sets.get(key, {}))

    def zadd(self, key, mapping):
        self._sets.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        pass

    def ping(self):
        return True


class _FakeRagPipeline:
    def __init__(self, events: list[tuple[str, dict]]) -> None:
        self._events = events
        self.calls: list[dict] = []

    async def answer_stream(self, message, session_id, filters=None):
        self.calls.append({"message": message, "session_id": session_id, "filters": filters})
        for event in self._events:
            yield event


class _OkPingable:
    def ping(self):
        return True


class _FailingPingable:
    def ping(self):
        raise ConnectionError("unreachable")


class _FakeIngestionPipeline:
    def __init__(self) -> None:
        self.run_calls: list[list] = []

    async def run(self, loaders):
        self.run_calls.append(loaders)

        class _Stats:
            chunks = 3
            documents = 1

        return _Stats()


class _FakeScheduler:
    running = True


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()
    if hasattr(app.state, "wiki_scheduler"):
        del app.state.wiki_scheduler


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_streams_session_tokens_citations_and_done(client):
    fake_pipeline = _FakeRagPipeline(
        [
            ("token", {"text": "Restart "}),
            ("token", {"text": "the pods."}),
            ("citations", {"citations": []}),
            ("done", {}),
        ]
    )
    app.dependency_overrides[get_redis_client] = lambda: _FakeRedis()
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_pipeline

    response = client.post("/chat", json={"message": "why is it down", "session_id": "sess-1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert 'event: session\ndata: {"session_id": "sess-1"}' in body
    assert 'event: token\ndata: {"text": "Restart "}' in body
    assert "event: done\ndata: {}" in body
    assert fake_pipeline.calls[0]["message"] == "why is it down"
    assert fake_pipeline.calls[0]["session_id"] == "sess-1"


def test_chat_generates_a_session_id_when_none_supplied(client):
    fake_pipeline = _FakeRagPipeline([("done", {})])
    app.dependency_overrides[get_redis_client] = lambda: _FakeRedis()
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_pipeline

    response = client.post("/chat", json={"message": "hello"})

    generated_id = fake_pipeline.calls[0]["session_id"]
    assert generated_id
    assert f'"session_id": "{generated_id}"' in response.text


def test_chat_passes_filters_through_to_the_pipeline(client):
    fake_pipeline = _FakeRagPipeline([("done", {})])
    app.dependency_overrides[get_redis_client] = lambda: _FakeRedis()
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_pipeline

    client.post(
        "/chat",
        json={
            "message": "hello",
            "session_id": "sess-1",
            "filters": {"service": "payment-service"},
        },
    )

    assert fake_pipeline.calls[0]["filters"] == {"service": "payment-service"}


def test_chat_rejects_the_11th_request_within_a_minute(client):
    fake_redis = _FakeRedis()
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.dependency_overrides[get_rag_pipeline] = lambda: _FakeRagPipeline([("done", {})])

    for _ in range(10):
        response = client.post("/chat", json={"message": "hi", "session_id": "sess-1"})
        assert response.status_code == 200

    response = client.post("/chat", json={"message": "hi", "session_id": "sess-1"})
    assert response.status_code == 429


def _mock_health_factories(monkeypatch, **overrides):
    # health() calls the get_* factories directly (not via Depends()) so that a
    # construction failure - not just a failed .ping() - is also caught. That means
    # app.dependency_overrides no longer reaches it; tests patch api.main's names
    # directly instead. Defaults to a healthy factory for anything not overridden.
    for name in ("get_redis_client", "get_vector_retriever", "get_keyword_retriever", "get_graph_retriever"):
        monkeypatch.setattr(f"api.main.{name}", overrides.get(name, lambda: _OkPingable()))


def test_health_reports_ok_when_every_service_responds(client, monkeypatch):
    _mock_health_factories(monkeypatch)

    async def fake_index_exists():
        return True

    monkeypatch.setattr("api.main.wiki_index_exists", fake_index_exists)
    app.state.wiki_scheduler = _FakeScheduler()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"] == {
        "redis": "ok",
        "chromadb": "ok",
        "elasticsearch": "ok",
        "neo4j": "ok",
        "wiki_index": "ok",
        "wiki_scheduler": "ok",
    }


def test_health_reports_degraded_when_one_service_fails(client, monkeypatch):
    _mock_health_factories(monkeypatch, get_graph_retriever=lambda: _FailingPingable())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["neo4j"]


def test_health_reports_degraded_instead_of_500_when_a_dependency_cannot_even_be_constructed(
    client, monkeypatch
):
    # Regression case: ChromaVectorRetriever's client eagerly connects, so
    # get_vector_retriever() can raise before health() ever gets a chance to try
    # a .ping() at all. Depends(get_vector_retriever) would 500 the whole
    # request here; calling the factory inside health()'s own try/except must not.
    def failing_factory():
        raise ValueError("Could not connect to a Chroma server. Are you sure it is running?")

    _mock_health_factories(monkeypatch, get_vector_retriever=failing_factory)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["chromadb"]


def test_health_reports_wiki_index_missing_as_degraded(client, monkeypatch):
    _mock_health_factories(monkeypatch)

    async def fake_index_missing():
        return False

    monkeypatch.setattr("api.main.wiki_index_exists", fake_index_missing)
    app.state.wiki_scheduler = _FakeScheduler()

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["wiki_index"]


def test_health_reports_scheduler_not_running_as_degraded(client, monkeypatch):
    _mock_health_factories(monkeypatch)

    async def fake_index_exists():
        return True

    monkeypatch.setattr("api.main.wiki_index_exists", fake_index_exists)
    # No app.state.wiki_scheduler set - mirrors reality when lifespan hasn't run.

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["wiki_scheduler"]


def test_ingest_schedules_a_background_task_and_returns_202(client):
    fake_pipeline = _FakeIngestionPipeline()
    app.dependency_overrides[get_ingestion_pipeline] = lambda: fake_pipeline

    response = client.post("/ingest", json={"source_type": "local", "path": "/tmp/runbooks"})

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "source_type": "local"}
    assert len(fake_pipeline.run_calls) == 1


def test_ingest_requires_path_for_local_source(client):
    app.dependency_overrides[get_ingestion_pipeline] = lambda: _FakeIngestionPipeline()

    response = client.post("/ingest", json={"source_type": "local"})

    assert response.status_code == 422


def test_graph_explore_returns_nodes_and_edges(client):
    class _FakeGraphRetriever:
        def explore_subgraph(self, entity_name):
            assert entity_name == "auth-service"
            return {
                "nodes": [{"id": "auth-service", "name": "auth-service", "labels": ["SERVICE"]}],
                "edges": [],
            }

    app.dependency_overrides[get_graph_retriever] = lambda: _FakeGraphRetriever()

    response = client.get("/graph/explore", params={"entity_name": "auth-service"})

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["id"] == "auth-service"
    assert body["edges"] == []


def test_lifespan_creates_wiki_index_and_starts_invalidation_scheduler(monkeypatch):
    create_calls: list[bool] = []

    async def fake_create_wiki_index():
        create_calls.append(True)

    monkeypatch.setattr("api.main.create_wiki_index", fake_create_wiki_index)

    with TestClient(app):
        assert create_calls == [True]

        scheduler = app.state.wiki_scheduler
        assert scheduler.running is True

        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].func is invalidate_stale
        assert jobs[0].trigger.interval == timedelta(hours=WIKI_INVALIDATION_INTERVAL_HOURS)

    # shutdown() runs when the context exits
    assert scheduler.running is False


def test_lifespan_starts_up_even_if_the_wiki_index_cannot_be_created(monkeypatch):
    # Elasticsearch not being reachable at boot must not prevent the whole API
    # from starting - only the wiki layer degrades, checked separately by /health.
    async def failing_create_wiki_index():
        raise ConnectionError("elasticsearch unreachable")

    monkeypatch.setattr("api.main.create_wiki_index", failing_create_wiki_index)

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_get_rag_pipeline_omits_a_retriever_that_fails_to_construct(monkeypatch):
    def failing_vector_retriever():
        raise ConnectionError("chromadb unreachable")

    monkeypatch.setattr("api.main.get_vector_retriever", failing_vector_retriever)
    monkeypatch.setattr("api.main.get_keyword_retriever", lambda: object())
    monkeypatch.setattr("api.main.get_graph_retriever", lambda: object())
    monkeypatch.setattr("api.main.get_reranker", lambda: object())
    monkeypatch.setattr("api.main.get_answer_generator", lambda: object())
    monkeypatch.setattr("api.main.get_redis_client", lambda: object())

    pipeline = get_rag_pipeline(settings=get_settings())

    assert "vector" not in pipeline._retrievers
    assert "keyword" in pipeline._retrievers
    assert "graph" in pipeline._retrievers


def test_get_rag_pipeline_falls_back_to_passthrough_reranker(monkeypatch):
    def failing_reranker():
        raise OSError("could not download the cross-encoder model")

    monkeypatch.setattr("api.main.get_vector_retriever", lambda: object())
    monkeypatch.setattr("api.main.get_keyword_retriever", lambda: object())
    monkeypatch.setattr("api.main.get_graph_retriever", lambda: object())
    monkeypatch.setattr("api.main.get_reranker", failing_reranker)
    monkeypatch.setattr("api.main.get_answer_generator", lambda: object())
    monkeypatch.setattr("api.main.get_redis_client", lambda: object())

    pipeline = get_rag_pipeline(settings=get_settings())

    assert isinstance(pipeline._reranker, PassthroughReranker)
