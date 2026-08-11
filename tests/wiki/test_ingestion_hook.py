from __future__ import annotations

import asyncio

import pytest

import ingestion.pipeline as pipeline_module
from ingestion.embedders.base import BaseEmbedder
from ingestion.graph_builder import GraphBuilder
from ingestion.loaders.base import BaseLoader
from ingestion.models import Chunk, Document, SourceType
from ingestion.pipeline import IngestionPipeline


class _FakeEmbedder(BaseEmbedder):
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeChunker:
    """Splits on sentence boundaries instead of tokenizing - avoids TextChunker's
    tiktoken dependency, which the tests have no need to exercise."""

    def chunk(self, document: Document) -> list[Chunk]:
        parts = [part.strip() for part in document.content.split(".") if part.strip()]
        return [
            Chunk(
                id=f"{document.id}::{index}",
                doc_id=document.id,
                chunk_index=index,
                content=part,
                title=document.title,
                source_type=document.source_type,
                url=document.url,
                incident_date=document.incident_date,
                severity=document.severity,
                service_tags=document.service_tags,
                extra=document.extra,
            )
            for index, part in enumerate(parts)
        ]


class _FakeCollection:
    def upsert(self, **kwargs) -> None:
        pass


class _FakeChromaClient:
    def get_or_create_collection(self, name, metadata=None):
        return _FakeCollection()


class _FakeIndices:
    def exists(self, index) -> bool:
        return True

    def create(self, index, mappings) -> None:
        pass


class _FakeEsClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()


class _NoOpGraphBuilder(GraphBuilder):
    def __init__(self) -> None:  # intentionally skip the real GraphBuilder.__init__
        pass

    async def build(self, chunks) -> None:
        pass


class _FakeLoader(BaseLoader):
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def load(self) -> list[Document]:
        return self._documents


def make_document(doc_id: str, title: str, content: str, **kwargs) -> Document:
    kwargs.setdefault("source_type", SourceType.MARKDOWN)
    return Document(id=doc_id, title=title, content=content, **kwargs)


@pytest.fixture(autouse=True)
def _fake_chunker(monkeypatch):
    monkeypatch.setattr(pipeline_module, "TextChunker", lambda **kwargs: _FakeChunker())


@pytest.fixture(autouse=True)
def _fake_bulk(monkeypatch):
    monkeypatch.setattr(
        pipeline_module.helpers, "bulk", lambda client, actions: (len(list(actions)), [])
    )


@pytest.fixture
def distill_calls(monkeypatch):
    calls: list[list[dict]] = []

    async def fake_distill(chunks: list[dict]):
        calls.append(chunks)

    monkeypatch.setattr("wiki.distiller.distill", fake_distill)
    return calls


def make_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        chroma_client=_FakeChromaClient(),
        es_client=_FakeEsClient(),
        neo4j_driver=object(),
        embedder=_FakeEmbedder(),
        graph_builder=_NoOpGraphBuilder(),
    )


async def _drain_background_tasks(pipeline: IngestionPipeline) -> None:
    # Distillation is fired via asyncio.create_task and not awaited by run(), so
    # tests must yield back to the loop for those tasks to actually execute.
    tasks = list(pipeline._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


async def test_run_queues_one_distillation_call_per_unique_doc_id(distill_calls):
    pipeline = make_pipeline()
    documents = [
        make_document("doc-1", "Payment outage", "ERR-503 payment failure. " * 3),
        make_document("doc-2", "Auth outage", "ERR-401 auth failure."),
    ]
    loader = _FakeLoader(documents)

    stats = await pipeline.run([loader])
    await _drain_background_tasks(pipeline)

    assert stats.documents == 2
    doc_ids_seen = {chunks[0]["doc_id"] for chunks in distill_calls}
    assert doc_ids_seen == {"doc-1", "doc-2"}
    assert len(distill_calls) == 2


async def test_all_chunks_for_a_doc_id_are_passed_to_a_single_distill_call(distill_calls):
    pipeline = make_pipeline()
    # Three sentences -> three chunks from _FakeChunker, all belonging to doc-1.
    documents = [
        make_document(
            "doc-1", "Payment outage", "First sentence. Second sentence. Third sentence."
        )
    ]
    loader = _FakeLoader(documents)

    await pipeline.run([loader])
    await _drain_background_tasks(pipeline)

    assert len(distill_calls) == 1
    chunks_sent = distill_calls[0]
    assert len(chunks_sent) == 3
    assert all(chunk["doc_id"] == "doc-1" for chunk in chunks_sent)


async def test_run_does_not_queue_distillation_when_there_are_no_chunks(distill_calls):
    pipeline = make_pipeline()

    stats = await pipeline.run([_FakeLoader([])])

    assert stats.chunks == 0
    assert distill_calls == []


async def test_distiller_chunk_metadata_maps_source_type_and_status():
    pipeline = make_pipeline()
    pagerduty_doc = make_document(
        "pd-1",
        "Payment outage",
        "payment-service down.",
        source_type=SourceType.PAGERDUTY,
        extra={"status": "resolved"},
    )
    chunks = pipeline._chunker.chunk(pagerduty_doc)

    distiller_chunk = pipeline._to_distiller_chunk(chunks[0])

    assert distiller_chunk["doc_id"] == "pd-1"
    assert distiller_chunk["metadata"] == {"source_type": "incident", "status": "resolved"}
