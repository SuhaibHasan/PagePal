from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import anthropic
import httpx
import pytest
from elasticsearch import NotFoundError

from wiki import retriever
from wiki.schema import WikiEntry

VALID_SIGNALS = {
    "services": ["payment-service"],
    "error_codes": ["ERR-503"],
    "tags": ["payment-service"],
    "intent": "diagnosis",
}


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error

    async def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error=None) -> None:
        self.messages = _FakeMessages(response=response, error=error)


def make_signals_client(signals: dict = VALID_SIGNALS) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(response=_FakeMessage([_FakeTextBlock(json.dumps(signals))]))


class _FakeEsClient:
    def __init__(self, response=None, error=None) -> None:
        self._response = response if response is not None else {"hits": {"hits": []}}
        self._error = error
        self.search_calls: list[dict] = []

    def search(self, index, query, size):
        self.search_calls.append({"index": index, "query": query, "size": size})
        if self._error:
            raise self._error
        return self._response


def make_entry_hit(score: float, entry_id: str = "entry-1", **overrides) -> dict:
    now = datetime.now(UTC)
    defaults = {
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "confidence": 0.9,
        "affected_services": ["payment-service"],
        "related_error_codes": ["ERR-503"],
        "tags": ["payment-service"],
        "created_at": now,
        "last_updated": now,
        "last_validated": now,
    }
    defaults.update(overrides)
    entry = WikiEntry(id=entry_id, **defaults)
    source = entry.model_dump(mode="json", exclude={"id"})
    return {"_id": entry.id, "_score": score, "_source": source}


def search_response(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


@pytest.fixture(autouse=True)
def _capture_increments(monkeypatch):
    calls: list[str] = []

    async def fake_increment(entry_id: str) -> None:
        calls.append(entry_id)

    monkeypatch.setattr(retriever, "increment_hit_count", fake_increment)
    return calls


def _use_clients(monkeypatch, anthropic_client, es_client) -> None:
    monkeypatch.setattr(retriever, "get_async_anthropic_client", lambda: anthropic_client)
    monkeypatch.setattr(retriever, "get_elasticsearch_client", lambda: es_client)


async def _drain_background_tasks() -> None:
    tasks = list(retriever._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


async def test_returns_a_high_confidence_fresh_entry(monkeypatch, _capture_increments):
    hit = make_entry_hit(score=9.0)  # 9.0 / 10.0 = 0.9, above the 0.75 threshold
    es_client = _FakeEsClient(response=search_response([hit]))
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("why are payments failing")
    await _drain_background_tasks()

    assert isinstance(result, WikiEntry)
    assert result.id == "entry-1"
    assert result.title == hit["_source"]["title"]
    assert es_client.search_calls[0]["index"] == "wiki_entries"
    assert es_client.search_calls[0]["size"] == 1


async def test_stale_entry_returns_none(monkeypatch, _capture_increments):
    stale_time = datetime.now(UTC).replace(year=datetime.now(UTC).year - 1)
    hit = make_entry_hit(score=9.0, last_updated=stale_time, ttl_days=30)
    es_client = _FakeEsClient(response=search_response([hit]))
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("why are payments failing")

    assert result is None
    assert _capture_increments == []


async def test_low_normalized_score_returns_none(monkeypatch, _capture_increments):
    hit = make_entry_hit(score=5.0)  # 5.0 / 10.0 = 0.5, below the 0.75 threshold
    es_client = _FakeEsClient(response=search_response([hit]))
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("why are payments failing")

    assert result is None
    assert _capture_increments == []


async def test_hit_count_is_incremented_for_a_returned_entry(monkeypatch, _capture_increments):
    hit = make_entry_hit(score=9.0, entry_id="entry-42")
    es_client = _FakeEsClient(response=search_response([hit]))
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("why are payments failing")
    await _drain_background_tasks()

    assert result is not None
    assert _capture_increments == ["entry-42"]


async def test_no_hits_returns_none(monkeypatch, _capture_increments):
    es_client = _FakeEsClient(response=search_response([]))
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("some totally unrelated query")

    assert result is None
    assert _capture_increments == []


async def test_missing_wiki_index_returns_none(monkeypatch, _capture_increments):
    error = NotFoundError.__new__(NotFoundError)  # bypass __init__'s elastic_transport args
    es_client = _FakeEsClient(error=error)
    _use_clients(monkeypatch, make_signals_client(), es_client)

    result = await retriever.retrieve("why are payments failing")

    assert result is None
    assert _capture_increments == []


async def test_signal_extraction_falls_back_gracefully_on_api_error(monkeypatch, _capture_increments):
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    anthropic_client = _FakeAnthropicClient(error=error)
    hit = make_entry_hit(score=9.0)
    es_client = _FakeEsClient(response=search_response([hit]))
    _use_clients(monkeypatch, anthropic_client, es_client)

    result = await retriever.retrieve("why are payments failing")
    await _drain_background_tasks()

    assert result is not None
    sent_query = es_client.search_calls[0]["query"]
    should_clauses = sent_query["bool"]["should"]
    # With no extracted signals, only the raw-query multi_match clause is present.
    assert len(should_clauses) == 1
    assert should_clauses[0]["multi_match"]["boost"] == retriever.TEXT_BOOST


async def test_build_query_includes_boosted_should_clauses_and_confidence_filter():
    signals = retriever.QuerySignals(
        services=["payment-service"], error_codes=["ERR-503"], tags=["postgres"]
    )

    query = retriever._build_query("payments down", signals)

    should = query["bool"]["should"]
    assert {"terms": {"affected_services": ["payment-service"], "boost": 4.0}} in should
    assert {"terms": {"related_error_codes": ["ERR-503"], "boost": 3.0}} in should
    assert {"terms": {"tags": ["postgres"], "boost": 2.0}} in should
    assert {
        "multi_match": {"query": "payments down", "fields": ["title", "summary"], "boost": 1.0}
    } in should
    assert query["bool"]["filter"] == [{"range": {"confidence": {"gte": 0.5}}}]
