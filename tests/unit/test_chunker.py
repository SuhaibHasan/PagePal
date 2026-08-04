import pytest
import tiktoken

from ingestion.chunkers.text_chunker import TextChunker
from ingestion.models import Document, SourceType

def make_document(word_count: int, **overrides) -> Document:
    content = " ".join(f"word{i}" for i in range(word_count))
    defaults = {"id": "doc-1", "title": "Test Doc", "content": content, "source_type": SourceType.MARKDOWN}
    defaults.update(overrides)
    return Document(**defaults)


def test_chunk_respects_chunk_size():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    document = make_document(500)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    encoding = tiktoken.get_encoding("cl100k_base")
    for chunk in chunks:
        assert len(encoding.encode(chunk.content)) <= 50


def test_chunks_overlap_between_consecutive_windows():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    document = make_document(100)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    first_words = set(chunks[0].content.split())
    second_words = set(chunks[1].content.split())
    assert first_words & second_words


def test_empty_document_produces_no_chunks():
    chunker = TextChunker()
    document = make_document(0, content="")

    assert chunker.chunk(document) == []


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_chunk_preserves_document_metadata_for_idempotent_ids():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    document = Document(
        id="doc-3",
        title="Payment Outage",
        content=" ".join(f"word{i}" for i in range(50)),
        source_type=SourceType.PAGERDUTY,
        url="https://example.pagerduty.com/incidents/ABC123",
        severity="high",
        service_tags=["payment-service"],
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.id == f"doc-3::{index}"
        assert chunk.doc_id == "doc-3"
        assert chunk.chunk_index == index
        assert chunk.title == "Payment Outage"
        assert chunk.source_type == SourceType.PAGERDUTY
        assert chunk.url == document.url
        assert chunk.severity == "high"
        assert chunk.service_tags == ["payment-service"]
