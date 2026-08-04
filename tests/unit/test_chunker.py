import pytest

from ingestion.chunkers.text_chunker import TextChunker
from ingestion.models import Document


def make_document(word_count: int) -> Document:
    content = " ".join(f"word{i}" for i in range(word_count))
    return Document(id="doc-1", source="test.md", content=content)


def test_chunk_respects_chunk_size():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    document = make_document(500)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    for chunk in chunks:
        token_count = len(chunker._encoding.encode(chunk.content))
        assert token_count <= 50


def test_chunk_overlap_repeats_tail_tokens():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    document = make_document(100)

    chunks = chunker.chunk(document)

    first_tail = chunker._encoding.encode(chunks[0].content)[-5:]
    second_head = chunker._encoding.encode(chunks[1].content)[:5]
    assert first_tail == second_head


def test_empty_document_produces_no_chunks():
    chunker = TextChunker()
    document = Document(id="doc-2", source="empty.md", content="")

    assert chunker.chunk(document) == []


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_chunk_metadata_includes_source_and_index():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    document = make_document(100)

    chunks = chunker.chunk(document)

    assert chunks[0].metadata["source"] == "test.md"
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
