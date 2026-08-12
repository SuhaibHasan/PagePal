from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from wiki import writer
from wiki.schema import WikiEntry

VALID_PAYLOAD = {
    "title": "payment-service ERR-503 under connection pool exhaustion",
    "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
    "root_cause": "A deployment reduced the connection pool size below traffic needs.",
    "resolution_steps": ["Restart pods", "Raise the connection pool size"],
    "affected_services": ["payment-service", "postgres-primary"],
    "related_error_codes": ["ERR-503"],
    "tags": ["payment-service", "postgres"],
    "severity_pattern": "high",
    "avg_resolution_time": "20 minutes",
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
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error=None) -> None:
        self.messages = _FakeMessages(response=response, error=error)


def make_client(text: str) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(response=_FakeMessage([_FakeTextBlock(text)]))


@pytest.fixture
def upsert_calls(monkeypatch):
    calls: list[WikiEntry] = []

    async def fake_upsert(entry: WikiEntry) -> None:
        calls.append(entry)

    monkeypatch.setattr(writer, "upsert_wiki_entry", fake_upsert)
    return calls


def _use_client(monkeypatch, client) -> None:
    monkeypatch.setattr(writer, "get_async_anthropic_client", lambda: client)


def _use_existing_entry(monkeypatch, entry: WikiEntry | None) -> None:
    async def fake_get(entry_id: str):
        return entry

    monkeypatch.setattr(writer, "get_wiki_entry", fake_get)


async def test_valid_answer_creates_a_new_entry_at_confidence_0_6(monkeypatch, upsert_calls):
    client = make_client(json.dumps(VALID_PAYLOAD))
    _use_client(monkeypatch, client)
    _use_existing_entry(monkeypatch, None)
    citations = [{"title": "Runbook A"}, {"title": "Runbook B"}, {"title": None}]

    await writer.upsert_from_rag("why did payments fail", "restart the pods", citations)

    assert len(upsert_calls) == 1
    entry = upsert_calls[0]
    assert entry.title == VALID_PAYLOAD["title"]
    assert entry.confidence == 0.6
    assert entry.source_refs == ["Runbook A", "Runbook B"]

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert sent_content == "Question: why did payments fail\n\nAnswer: restart the pods"
    assert client.messages.calls[0]["system"] == writer.SYSTEM_PROMPT
    assert client.messages.calls[0]["max_tokens"] == 800


async def test_existing_entry_is_merged_and_confidence_bumped(monkeypatch, upsert_calls):
    client = make_client(json.dumps(VALID_PAYLOAD))
    _use_client(monkeypatch, client)

    old_time = datetime.now(UTC) - timedelta(days=10)
    existing = WikiEntry(
        id="existing-id",
        title="Original title",
        summary="Original summary",
        confidence=0.8,
        resolution_steps=["Restart pods"],
        affected_services=["payment-service"],
        related_error_codes=["ERR-500"],
        tags=["payment-service"],
        source_refs=["Old Runbook"],
        created_at=old_time,
        last_updated=old_time,
        last_validated=old_time,
    )
    _use_existing_entry(monkeypatch, existing)

    await writer.upsert_from_rag("why did payments fail", "restart the pods", [])

    assert len(upsert_calls) == 1
    entry = upsert_calls[0]
    assert entry.id == "existing-id"
    # Merged list fields: existing items first, new ones appended, no duplicates.
    assert entry.resolution_steps == ["Restart pods", "Raise the connection pool size"]
    assert entry.affected_services == ["payment-service", "postgres-primary"]
    assert entry.related_error_codes == ["ERR-500", "ERR-503"]
    assert entry.tags == ["payment-service", "postgres"]
    assert entry.confidence == pytest.approx(0.85)
    assert entry.last_updated > old_time
    # Fields not called out for merging stay as the existing entry's values.
    assert entry.title == "Original title"
    assert entry.summary == "Original summary"
    assert entry.source_refs == ["Old Runbook"]
    assert entry.created_at == old_time


async def test_related_error_codes_from_a_second_rag_answer_are_merged_and_deduplicated(
    monkeypatch, upsert_calls
):
    # Simulates a second RAG answer distilling to a payload that repeats an
    # error code the entry already has (ERR-503) alongside a genuinely new one.
    second_answer_payload = {**VALID_PAYLOAD, "related_error_codes": ["ERR-503", "ERR-429"]}
    client = make_client(json.dumps(second_answer_payload))
    _use_client(monkeypatch, client)

    existing = WikiEntry(
        id="existing-id",
        title="Original title",
        summary="Original summary",
        confidence=0.8,
        related_error_codes=["ERR-503"],
        created_at=datetime.now(UTC),
        last_updated=datetime.now(UTC),
        last_validated=datetime.now(UTC),
    )
    _use_existing_entry(monkeypatch, existing)

    await writer.upsert_from_rag("why did payments fail again", "raise the pool size", [])

    entry = upsert_calls[0]
    assert entry.related_error_codes == ["ERR-503", "ERR-429"]


async def test_confidence_bump_is_capped_at_0_95(monkeypatch, upsert_calls):
    client = make_client(json.dumps(VALID_PAYLOAD))
    _use_client(monkeypatch, client)
    now = datetime.now(UTC)
    existing = WikiEntry(
        id="existing-id",
        title="Original title",
        summary="Original summary",
        confidence=0.93,
        created_at=now,
        last_updated=now,
        last_validated=now,
    )
    _use_existing_entry(monkeypatch, existing)

    await writer.upsert_from_rag("q", "a", [])

    assert upsert_calls[0].confidence == pytest.approx(0.95)


async def test_vague_answer_returns_early_without_upserting(monkeypatch, upsert_calls):
    client = make_client("null")
    _use_client(monkeypatch, client)
    get_called = False

    async def fake_get(entry_id: str):
        nonlocal get_called
        get_called = True

    monkeypatch.setattr(writer, "get_wiki_entry", fake_get)

    await writer.upsert_from_rag("what's the weather", "I don't know", [])

    assert upsert_calls == []
    assert get_called is False


async def test_exception_from_the_llm_call_never_propagates(monkeypatch, upsert_calls):
    client = _FakeAnthropicClient(error=RuntimeError("boom"))
    _use_client(monkeypatch, client)

    await writer.upsert_from_rag("why did payments fail", "restart the pods", [])

    assert upsert_calls == []


async def test_exception_from_invalid_json_never_propagates(monkeypatch, upsert_calls):
    client = make_client("this is not json")
    _use_client(monkeypatch, client)

    await writer.upsert_from_rag("why did payments fail", "restart the pods", [])

    assert upsert_calls == []


async def test_exception_from_get_wiki_entry_never_propagates(monkeypatch, upsert_calls):
    client = make_client(json.dumps(VALID_PAYLOAD))
    _use_client(monkeypatch, client)

    async def failing_get(entry_id: str):
        raise ConnectionError("elasticsearch unreachable")

    monkeypatch.setattr(writer, "get_wiki_entry", failing_get)

    await writer.upsert_from_rag("why did payments fail", "restart the pods", [])

    assert upsert_calls == []
