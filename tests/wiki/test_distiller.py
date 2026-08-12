from __future__ import annotations

import hashlib
import json

import anthropic
import httpx
import pytest
import tiktoken

from wiki import distiller
from wiki.schema import WikiEntry

VALID_PAYLOAD = {
    "title": "payment-service ERR-503 under connection pool exhaustion",
    "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
    "root_cause": "A deployment reduced the connection pool size below traffic needs.",
    "resolution_steps": ["Restart payment-service pods", "Raise the connection pool size"],
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


def make_chunk(
    doc_id: str = "doc-1",
    content: str = "The payment-service returned ERR-503 due to pool exhaustion.",
    source_type: str = "incident",
    status: str | None = "resolved",
) -> dict:
    return {
        "doc_id": doc_id,
        "content": content,
        "metadata": {"source_type": source_type, "status": status},
    }


@pytest.fixture(autouse=True)
def _capture_upserts(monkeypatch):
    calls: list[WikiEntry] = []

    async def fake_upsert(entry: WikiEntry) -> None:
        calls.append(entry)

    monkeypatch.setattr(distiller, "upsert_wiki_entry", fake_upsert)
    return calls


def _use_client(monkeypatch, client) -> None:
    monkeypatch.setattr(distiller, "get_async_anthropic_client", lambda: client)


async def test_distill_produces_a_wiki_entry_from_a_valid_response(monkeypatch, _capture_upserts):
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeTextBlock(json.dumps(VALID_PAYLOAD))])
    )
    _use_client(monkeypatch, client)
    chunk = make_chunk()

    entry = await distiller.distill([chunk])

    assert isinstance(entry, WikiEntry)
    assert entry.title == VALID_PAYLOAD["title"]
    assert entry.summary == VALID_PAYLOAD["summary"]
    assert entry.root_cause == VALID_PAYLOAD["root_cause"]
    assert entry.resolution_steps == VALID_PAYLOAD["resolution_steps"]
    assert entry.affected_services == VALID_PAYLOAD["affected_services"]
    assert entry.related_error_codes == VALID_PAYLOAD["related_error_codes"]
    assert entry.tags == VALID_PAYLOAD["tags"]
    assert entry.severity_pattern == VALID_PAYLOAD["severity_pattern"]
    assert entry.avg_resolution_time == VALID_PAYLOAD["avg_resolution_time"]
    assert entry.source_refs == ["doc-1"]
    assert entry.hit_count == 0
    assert entry.ttl_days == 30
    assert entry.confidence == 0.9  # incident + resolved
    expected_hash = hashlib.sha256(distiller.concatenate_and_trim([chunk]).encode()).hexdigest()
    assert entry.source_content_hash == expected_hash

    # upsert_wiki_entry was called with the same entry that was returned
    assert _capture_upserts == [entry]

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert chunk["content"] in sent_content
    assert client.messages.calls[0]["system"] == distiller.SYSTEM_PROMPT


async def test_distill_returns_none_when_the_model_returns_the_string_null(
    monkeypatch, _capture_upserts
):
    client = _FakeAnthropicClient(response=_FakeMessage([_FakeTextBlock("null")]))
    _use_client(monkeypatch, client)

    entry = await distiller.distill([make_chunk()])

    assert entry is None
    assert _capture_upserts == []


async def test_distill_returns_none_on_api_error(monkeypatch, _capture_upserts):
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    client = _FakeAnthropicClient(error=error)
    _use_client(monkeypatch, client)

    entry = await distiller.distill([make_chunk()])

    assert entry is None
    assert _capture_upserts == []


async def test_distill_returns_none_on_invalid_json(monkeypatch, _capture_upserts):
    client = _FakeAnthropicClient(response=_FakeMessage([_FakeTextBlock("not valid json")]))
    _use_client(monkeypatch, client)

    entry = await distiller.distill([make_chunk()])

    assert entry is None
    assert _capture_upserts == []


async def test_distill_returns_none_when_required_fields_are_missing(monkeypatch, _capture_upserts):
    incomplete_payload = {"title": "Something broke"}  # missing summary, confidence inputs, etc.
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeTextBlock(json.dumps(incomplete_payload))])
    )
    _use_client(monkeypatch, client)

    entry = await distiller.distill([make_chunk()])

    assert entry is None
    assert _capture_upserts == []


async def test_distill_returns_none_for_an_empty_chunk_list(monkeypatch, _capture_upserts):
    entry = await distiller.distill([])

    assert entry is None
    assert _capture_upserts == []


@pytest.mark.parametrize(
    "source_type,status,expected_confidence",
    [
        ("incident", "resolved", 0.9),
        ("runbook", None, 0.85),
        ("jira", "resolved", 0.7),
        ("jira", "open", 0.5),
        ("confluence", None, 0.4),
        ("incident", "open", 0.4),
    ],
)
async def test_distill_assigns_confidence_by_source_type_and_status(
    monkeypatch, source_type, status, expected_confidence
):
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeTextBlock(json.dumps(VALID_PAYLOAD))])
    )
    _use_client(monkeypatch, client)
    chunk = make_chunk(source_type=source_type, status=status)

    entry = await distiller.distill([chunk])

    assert entry.confidence == expected_confidence


async def test_distill_trims_content_over_4000_tokens_before_calling_the_model(monkeypatch):
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeTextBlock(json.dumps(VALID_PAYLOAD))])
    )
    _use_client(monkeypatch, client)
    long_content = "payment-service ERR-503 outage details. " * 2000
    chunk = make_chunk(content=long_content)

    encoding = tiktoken.get_encoding(distiller.ENCODING_NAME)
    assert len(encoding.encode(long_content)) > distiller.MAX_CONTENT_TOKENS

    await distiller.distill([chunk])

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert len(encoding.encode(sent_content)) <= distiller.MAX_CONTENT_TOKENS
    assert len(sent_content) < len(long_content)


async def test_distill_source_refs_dedupes_doc_ids_preserving_order(monkeypatch, _capture_upserts):
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeTextBlock(json.dumps(VALID_PAYLOAD))])
    )
    _use_client(monkeypatch, client)
    chunks = [
        make_chunk(doc_id="doc-1", content="first chunk"),
        make_chunk(doc_id="doc-2", content="second chunk"),
        make_chunk(doc_id="doc-1", content="third chunk"),
    ]

    entry = await distiller.distill(chunks)

    assert entry.source_refs == ["doc-1", "doc-2"]
