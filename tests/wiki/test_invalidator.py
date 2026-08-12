from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from wiki import invalidator
from wiki.schema import WikiEntry


class _FakeEsClient:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.search_calls: list[dict] = []

    def search(self, index, query, size):
        self.search_calls.append({"index": index, "query": query, "size": size})
        return {"hits": {"hits": self._hits}}


class _FakeCollection:
    def __init__(self, chunks_by_doc_id: dict[str, list[dict]]) -> None:
        self._chunks_by_doc_id = chunks_by_doc_id
        self.get_calls: list[dict] = []

    def get(self, where, include):
        self.get_calls.append({"where": where, "include": include})
        rows = self._chunks_by_doc_id.get(where["doc_id"], [])
        return {
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }


def make_hit(entry: WikiEntry) -> dict:
    return {"_id": entry.id, "_score": 1.0, "_source": entry.model_dump(mode="json", exclude={"id"})}


def make_stale_entry(**overrides) -> WikiEntry:
    stale_time = datetime.now(UTC) - timedelta(days=45)
    defaults = {
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "confidence": 0.9,
        "source_refs": ["doc-1"],
        "created_at": stale_time,
        "last_updated": stale_time,
        "last_validated": stale_time,
        "ttl_days": 30,
    }
    defaults.update(overrides)
    return WikiEntry(**defaults)


def hash_of(*contents: str) -> str:
    return hashlib.sha256("\n\n".join(contents).encode()).hexdigest()


@pytest.fixture(autouse=True)
def _fake_concatenate(monkeypatch):
    # Avoids the real tiktoken-backed concatenate_and_trim (a network dependency);
    # a plain join is all invalidator.py's hash comparison needs to be exercised.
    monkeypatch.setattr(
        invalidator,
        "concatenate_and_trim",
        lambda chunks: "\n\n".join(chunk["content"] for chunk in chunks),
    )


@pytest.fixture
def upsert_calls(monkeypatch):
    calls: list[WikiEntry] = []

    async def fake_upsert(entry: WikiEntry) -> None:
        calls.append(entry)

    monkeypatch.setattr(invalidator, "upsert_wiki_entry", fake_upsert)
    return calls


@pytest.fixture
def distill_calls(monkeypatch):
    calls: list[list[dict]] = []

    async def fake_distill(chunks: list[dict]):
        calls.append(chunks)

    monkeypatch.setattr(invalidator, "distill", fake_distill)
    return calls


def _use_clients(monkeypatch, es_client, collection) -> None:
    monkeypatch.setattr(invalidator, "get_elasticsearch_client", lambda: es_client)
    monkeypatch.setattr(invalidator, "_get_collection", lambda: collection)


async def test_invalidate_stale_refreshes_an_entry_with_unchanged_content(
    monkeypatch, upsert_calls, distill_calls
):
    content = "payment-service pool exhausted"
    entry = make_stale_entry(source_content_hash=hash_of(content))
    es_client = _FakeEsClient([make_hit(entry)])
    collection = _FakeCollection(
        {"doc-1": [{"document": content, "metadata": {"source_type": "markdown"}}]}
    )
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 1, "redistilled": 0, "soft_deleted": 0}
    assert distill_calls == []
    assert len(upsert_calls) == 1
    refreshed = upsert_calls[0]
    assert refreshed.id == entry.id
    assert refreshed.confidence == entry.confidence
    assert refreshed.last_validated > entry.last_validated


async def test_invalidate_stale_redistills_an_entry_with_changed_content(
    monkeypatch, upsert_calls, distill_calls
):
    entry = make_stale_entry(source_content_hash=hash_of("the old content"))
    es_client = _FakeEsClient([make_hit(entry)])
    collection = _FakeCollection(
        {"doc-1": [{"document": "brand new content", "metadata": {"source_type": "markdown"}}]}
    )
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 0, "redistilled": 1, "soft_deleted": 0}
    assert upsert_calls == []  # distill() itself upserts; invalidator doesn't call it separately
    assert len(distill_calls) == 1
    assert distill_calls[0][0]["content"] == "brand new content"
    assert distill_calls[0][0]["doc_id"] == "doc-1"


async def test_invalidate_stale_redistills_a_legacy_entry_with_no_stored_hash(
    monkeypatch, upsert_calls, distill_calls
):
    entry = make_stale_entry(source_content_hash=None)
    es_client = _FakeEsClient([make_hit(entry)])
    collection = _FakeCollection(
        {"doc-1": [{"document": "some content", "metadata": {"source_type": "markdown"}}]}
    )
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 0, "redistilled": 1, "soft_deleted": 0}


async def test_invalidate_stale_soft_deletes_an_entry_whose_chunks_are_gone(
    monkeypatch, upsert_calls, distill_calls
):
    entry = make_stale_entry(source_content_hash=hash_of("gone now"))
    es_client = _FakeEsClient([make_hit(entry)])
    collection = _FakeCollection({})  # doc-1 no longer has any chunks
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 0, "redistilled": 0, "soft_deleted": 1}
    assert distill_calls == []
    assert len(upsert_calls) == 1
    assert upsert_calls[0].confidence == 0.0
    assert upsert_calls[0].id == entry.id


async def test_invalidate_stale_skips_entries_that_are_not_actually_stale(
    monkeypatch, upsert_calls, distill_calls
):
    fresh_entry = make_stale_entry(
        last_updated=datetime.now(UTC), last_validated=datetime.now(UTC)
    )
    es_client = _FakeEsClient([make_hit(fresh_entry)])
    collection = _FakeCollection({})
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 0, "redistilled": 0, "soft_deleted": 0}
    assert upsert_calls == []
    assert distill_calls == []


async def test_invalidate_stale_returns_zero_counts_when_nothing_is_stale(monkeypatch):
    es_client = _FakeEsClient([])
    collection = _FakeCollection({})
    _use_clients(monkeypatch, es_client, collection)

    counts = await invalidator.invalidate_stale()

    assert counts == {"refreshed": 0, "redistilled": 0, "soft_deleted": 0}


async def test_invalidate_stale_returns_zero_counts_when_es_is_unreachable(monkeypatch):
    class _FailingEsClient:
        def search(self, index, query, size):
            raise ConnectionError("elasticsearch unreachable")

    monkeypatch.setattr(invalidator, "get_elasticsearch_client", lambda: _FailingEsClient())
    monkeypatch.setattr(invalidator, "_get_collection", lambda: _FakeCollection({}))

    counts = await invalidator.invalidate_stale()  # must not raise

    assert counts == {"refreshed": 0, "redistilled": 0, "soft_deleted": 0}


async def test_invalidate_for_doc_triggers_the_distiller_when_entries_reference_it(
    monkeypatch, distill_calls
):
    entry = make_stale_entry()
    es_client = _FakeEsClient([make_hit(entry)])
    collection = _FakeCollection(
        {"doc-1": [{"document": "updated doc-1 content", "metadata": {"source_type": "jira"}}]}
    )
    _use_clients(monkeypatch, es_client, collection)

    await invalidator.invalidate_for_doc("doc-1")

    assert es_client.search_calls[0]["query"] == {"term": {"source_refs": "doc-1"}}
    assert len(distill_calls) == 1
    assert distill_calls[0][0]["doc_id"] == "doc-1"
    assert distill_calls[0][0]["content"] == "updated doc-1 content"
    assert distill_calls[0][0]["metadata"]["source_type"] == "jira"


async def test_invalidate_for_doc_does_nothing_when_no_entries_reference_it(
    monkeypatch, distill_calls
):
    es_client = _FakeEsClient([])
    collection = _FakeCollection({"doc-1": [{"document": "content", "metadata": {}}]})
    _use_clients(monkeypatch, es_client, collection)

    await invalidator.invalidate_for_doc("doc-1")

    assert distill_calls == []


async def test_invalidate_for_doc_never_raises_on_es_failure(monkeypatch, distill_calls):
    class _FailingEsClient:
        def search(self, index, query, size):
            raise ConnectionError("elasticsearch unreachable")

    monkeypatch.setattr(invalidator, "get_elasticsearch_client", lambda: _FailingEsClient())

    await invalidator.invalidate_for_doc("doc-1")  # must not raise

    assert distill_calls == []
