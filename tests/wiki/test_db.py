from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from elasticsearch import Elasticsearch

from wiki import db
from wiki.schema import WikiEntry

try:
    from testcontainers.community.elasticsearch import ElasticSearchContainer
except ImportError:  # pragma: no cover
    ElasticSearchContainer = None

ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.15.3"


def make_entry(**overrides) -> WikiEntry:
    now = datetime.now(UTC)
    defaults = {
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "confidence": 0.85,
        "tags": ["payment-service"],
        "created_at": now,
        "last_updated": now,
        "last_validated": now,
    }
    defaults.update(overrides)
    return WikiEntry(**defaults)


@pytest.fixture(scope="module")
def es_container():
    if ElasticSearchContainer is None:
        pytest.skip("testcontainers elasticsearch support is not installed")

    try:
        container = ElasticSearchContainer(ES_IMAGE)
        container.with_env("discovery.type", "single-node")
        container.start()
    except Exception as exc:  # noqa: BLE001 - environment-dependent, not a code defect
        pytest.skip(f"Docker is unavailable to run the Elasticsearch test container: {exc}")

    yield container
    container.stop()


@pytest.fixture
def client(es_container) -> Iterator[Elasticsearch]:
    url = (
        f"http://{es_container.get_container_host_ip()}:"
        f"{es_container.get_exposed_port(es_container.port)}"
    )
    es_client = Elasticsearch(url)
    yield es_client
    es_client.indices.delete(index=db.WIKI_INDEX, ignore_unavailable=True)
    es_client.close()


@pytest.fixture(autouse=True)
def _use_test_client(monkeypatch, client):
    # wiki/db.py reuses api.dependencies.get_elasticsearch_client() rather than opening
    # its own connection; swapping that lookup is how tests point it at the container.
    monkeypatch.setattr(db, "get_elasticsearch_client", lambda: client)


async def test_create_wiki_index_creates_index_with_expected_mapping(client):
    await db.create_wiki_index()

    assert client.indices.exists(index=db.WIKI_INDEX)
    properties = client.indices.get_mapping(index=db.WIKI_INDEX)[db.WIKI_INDEX]["mappings"][
        "properties"
    ]
    assert properties["title"]["type"] == "text"
    assert properties["title"]["fields"]["keyword"]["type"] == "keyword"
    assert properties["tags"]["type"] == "keyword"
    assert properties["confidence"]["type"] == "float"
    assert properties["hit_count"]["type"] == "integer"
    assert properties["last_updated"]["type"] == "date"


async def test_create_wiki_index_is_idempotent(client):
    await db.create_wiki_index()
    await db.create_wiki_index()  # must not raise on a second call

    assert client.indices.exists(index=db.WIKI_INDEX)


async def test_upsert_then_get_round_trips_the_entry(client):
    await db.create_wiki_index()
    entry = make_entry()

    await db.upsert_wiki_entry(entry)
    fetched = await db.get_wiki_entry(entry.id)

    assert fetched is not None
    assert fetched.id == entry.id
    assert fetched.title == entry.title
    assert fetched.summary == entry.summary
    assert fetched.confidence == entry.confidence
    assert fetched.tags == entry.tags


async def test_upsert_updates_an_existing_entry_in_place(client):
    await db.create_wiki_index()
    entry = make_entry(hit_count=5)
    await db.upsert_wiki_entry(entry)

    updated = make_entry(id=entry.id, hit_count=99, summary="Updated summary.")
    await db.upsert_wiki_entry(updated)
    fetched = await db.get_wiki_entry(entry.id)

    assert fetched.hit_count == 99
    assert fetched.summary == "Updated summary."
    client.indices.refresh(index=db.WIKI_INDEX)  # count() isn't real-time like get()
    assert client.count(index=db.WIKI_INDEX)["count"] == 1


async def test_get_wiki_entry_returns_none_when_missing(client):
    await db.create_wiki_index()

    result = await db.get_wiki_entry("does-not-exist")

    assert result is None


async def test_increment_hit_count_increments_by_one(client):
    await db.create_wiki_index()
    entry = make_entry(hit_count=0)
    await db.upsert_wiki_entry(entry)

    await db.increment_hit_count(entry.id)
    fetched = await db.get_wiki_entry(entry.id)

    assert fetched.hit_count == 1


async def test_increment_hit_count_accumulates_across_calls(client):
    await db.create_wiki_index()
    entry = make_entry(hit_count=0)
    await db.upsert_wiki_entry(entry)

    await db.increment_hit_count(entry.id)
    await db.increment_hit_count(entry.id)
    await db.increment_hit_count(entry.id)
    fetched = await db.get_wiki_entry(entry.id)

    assert fetched.hit_count == 3
