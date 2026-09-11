from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from elasticsearch import NotFoundError

import wiki.db as db_module
from api.config import Settings, get_settings
from api.main import app
from wiki.schema import WikiEntry

BASE_URL = "http://test"


class _FakeEsClient:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.update_calls: list[dict] = []

    def seed(self, entry: WikiEntry) -> None:
        self.docs[entry.id] = entry.model_dump(mode="json", exclude={"id"})

    def get(self, index, id):
        if id not in self.docs:
            raise NotFoundError.__new__(NotFoundError)
        return {"_id": id, "_source": self.docs[id]}

    def update(self, index, id, doc=None, doc_as_upsert=None, script=None):
        self.update_calls.append({"id": id, "doc": doc, "script": script})
        if doc is not None:
            self.docs[id] = doc
        return {}

    def search(self, index, query, size):
        if query == {"match_all": {}}:
            matched_ids = list(self.docs.keys())
            scores = dict.fromkeys(matched_ids, 1.0)
        else:
            bool_query = query["bool"]
            needle = bool_query["must"][0]["multi_match"]["query"].lower()
            min_confidence = bool_query["filter"][0]["range"]["confidence"]["gte"]
            matched_ids = []
            scores = {}
            for doc_id, source in self.docs.items():
                haystack = " ".join(
                    [source.get("title", ""), source.get("summary", ""), *source.get("tags", [])]
                ).lower()
                if needle in haystack and source.get("confidence", 0.0) >= min_confidence:
                    matched_ids.append(doc_id)
                    scores[doc_id] = 5.0

        hits = [
            {"_id": doc_id, "_score": scores[doc_id], "_source": self.docs[doc_id]}
            for doc_id in matched_ids[:size]
        ]
        return {"hits": {"hits": hits}}


def make_entry(entry_id: str, **overrides) -> WikiEntry:
    now = datetime.now(UTC)
    defaults = {
        "id": entry_id,
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "tags": ["payment-service"],
        "confidence": 0.8,
        "hit_count": 0,
        "created_at": now,
        "last_updated": now,
        "last_validated": now,
        "ttl_days": 30,
    }
    defaults.update(overrides)
    return WikiEntry(**defaults)


@pytest.fixture
def fake_es(monkeypatch) -> _FakeEsClient:
    client = _FakeEsClient()
    monkeypatch.setattr(db_module, "get_elasticsearch_client", lambda: client)
    return client


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as async_client:
        yield async_client


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


async def test_search_returns_only_matching_and_confident_enough_results(fake_es, client):
    fake_es.seed(make_entry("payment-1", title="payment-service ERR-503", confidence=0.9))
    fake_es.seed(make_entry("payment-2", title="payment-service slow checkout", confidence=0.3))
    fake_es.seed(
        make_entry(
            "auth-1",
            title="auth-service login failures",
            summary="Users can't log in after a signing key rotation.",
            tags=["auth-service"],
            confidence=0.95,
        )
    )

    response = await client.get("/wiki/search", params={"q": "payment", "min_confidence": 0.5})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["entry"]["id"] == "payment-1"
    assert body[0]["score"] == 5.0


async def test_search_respects_default_min_confidence_of_0_5(fake_es, client):
    fake_es.seed(make_entry("payment-1", title="payment-service issue", confidence=0.4))

    response = await client.get("/wiki/search", params={"q": "payment"})

    assert response.json() == []


async def test_get_single_entry_returns_it(fake_es, client):
    fake_es.seed(make_entry("payment-1"))

    response = await client.get("/wiki/payment-1")

    assert response.status_code == 200
    assert response.json()["id"] == "payment-1"


async def test_get_single_entry_404s_when_missing(fake_es, client):
    response = await client.get("/wiki/does-not-exist")

    assert response.status_code == 404


async def test_validate_bumps_confidence_by_0_1_and_resets_last_validated(fake_es, client):
    stale_validation = datetime.now(UTC) - timedelta(days=10)
    fake_es.seed(make_entry("payment-1", confidence=0.7, last_validated=stale_validation))

    response = await client.post(
        "/wiki/payment-1/validate", headers={"X-Engineer-Id": "eng-42"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == pytest.approx(0.8)
    assert datetime.fromisoformat(body["last_validated"]) > stale_validation
    # persisted, not just returned
    assert fake_es.docs["payment-1"]["confidence"] == pytest.approx(0.8)


async def test_validate_caps_confidence_at_0_95(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.9))

    response = await client.post(
        "/wiki/payment-1/validate", headers={"X-Engineer-Id": "eng-42"}
    )

    assert response.json()["confidence"] == pytest.approx(0.95)


async def test_validate_requires_the_engineer_id_header(fake_es, client):
    fake_es.seed(make_entry("payment-1"))

    response = await client.post("/wiki/payment-1/validate")

    assert response.status_code == 422


async def test_validate_404s_when_entry_missing(fake_es, client):
    response = await client.post(
        "/wiki/does-not-exist/validate", headers={"X-Engineer-Id": "eng-42"}
    )

    assert response.status_code == 404


async def test_validate_rejects_requests_without_the_configured_api_key(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.7))
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret123")

    response = await client.post(
        "/wiki/payment-1/validate", headers={"X-Engineer-Id": "eng-42"}
    )

    assert response.status_code == 401


async def test_validate_accepts_the_correct_api_key(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.7))
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret123")

    response = await client.post(
        "/wiki/payment-1/validate",
        headers={"X-Engineer-Id": "eng-42", "X-API-Key": "secret123"},
    )

    assert response.status_code == 200


async def test_delete_sets_confidence_to_zero_without_removing_the_document(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.8))

    response = await client.delete("/wiki/payment-1")

    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0
    # still present in the index - a soft delete, not a real one
    assert "payment-1" in fake_es.docs
    assert fake_es.docs["payment-1"]["confidence"] == 0.0


async def test_delete_404s_when_entry_missing(fake_es, client):
    response = await client.delete("/wiki/does-not-exist")

    assert response.status_code == 404


async def test_delete_rejects_requests_without_the_configured_api_key(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.8))
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret123")

    response = await client.delete("/wiki/payment-1")

    assert response.status_code == 401
    # nothing was touched
    assert fake_es.docs["payment-1"]["confidence"] == 0.8


async def test_delete_accepts_the_correct_api_key(fake_es, client):
    fake_es.seed(make_entry("payment-1", confidence=0.8))
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret123")

    response = await client.delete("/wiki/payment-1", headers={"X-API-Key": "secret123"})

    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0


async def test_stats_returns_correct_counts(fake_es, client):
    now = datetime.now(UTC)
    stale_time = now - timedelta(days=45)

    fake_es.seed(make_entry("e1", confidence=0.9, hit_count=5, last_updated=now))  # high-conf
    fake_es.seed(make_entry("e2", confidence=0.6, hit_count=3, last_updated=now))
    fake_es.seed(
        make_entry("e3", confidence=0.85, hit_count=2, last_updated=stale_time)
    )  # stale + high-conf
    fake_es.seed(make_entry("e4", confidence=0.2, hit_count=0, last_updated=now))

    response = await client.get("/wiki/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["total_entries"] == 4
    assert body["avg_confidence"] == pytest.approx((0.9 + 0.6 + 0.85 + 0.2) / 4)
    assert body["total_hits"] == 10
    assert body["stale_count"] == 1
    assert body["high_confidence_count"] == 2


async def test_stats_with_no_entries(fake_es, client):
    response = await client.get("/wiki/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_entries": 0,
        "avg_confidence": 0.0,
        "total_hits": 0,
        "stale_count": 0,
        "high_confidence_count": 0,
    }
