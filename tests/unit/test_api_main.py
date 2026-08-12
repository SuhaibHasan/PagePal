from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_graph_retriever,
    get_ingestion_pipeline,
    get_keyword_retriever,
    get_redis_client,
    get_vector_retriever,
)
from api.main import WIKI_INVALIDATION_INTERVAL_HOURS, app, get_rag_pipeline
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


def test_health_reports_ok_when_every_service_responds(client, monkeypatch):
    app.dependency_overrides[get_redis_client] = lambda: _OkPingable()
    app.dependency_overrides[get_vector_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_keyword_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_graph_retriever] = lambda: _OkPingable()

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


def test_health_reports_degraded_when_one_service_fails(client):
    app.dependency_overrides[get_redis_client] = lambda: _OkPingable()
    app.dependency_overrides[get_vector_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_keyword_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_graph_retriever] = lambda: _FailingPingable()

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["neo4j"]


def test_health_reports_wiki_index_missing_as_degraded(client, monkeypatch):
    app.dependency_overrides[get_redis_client] = lambda: _OkPingable()
    app.dependency_overrides[get_vector_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_keyword_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_graph_retriever] = lambda: _OkPingable()

    async def fake_index_missing():
        return False

    monkeypatch.setattr("api.main.wiki_index_exists", fake_index_missing)
    app.state.wiki_scheduler = _FakeScheduler()

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["services"]["wiki_index"]


def test_health_reports_scheduler_not_running_as_degraded(client, monkeypatch):
    app.dependency_overrides[get_redis_client] = lambda: _OkPingable()
    app.dependency_overrides[get_vector_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_keyword_retriever] = lambda: _OkPingable()
    app.dependency_overrides[get_graph_retriever] = lambda: _OkPingable()

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
