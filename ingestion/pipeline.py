from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chromadb
from elasticsearch import Elasticsearch, helpers
from neo4j import AsyncDriver

from ingestion.chunkers.text_chunker import TextChunker
from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.embedder import SentenceTransformerEmbedder
from ingestion.graph_builder import GraphBuilder
from ingestion.loaders.base import BaseLoader
from ingestion.models import Chunk, Document

ES_INDEX_MAPPING = {
    "properties": {
        "content": {"type": "text"},
        "title": {"type": "text"},
        "tags": {"type": "keyword"},
        "severity": {"type": "keyword"},
        "service": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "url": {"type": "keyword"},
        "source_type": {"type": "keyword"},
        "incident_date": {"type": "date"},
    }
}


@dataclass
class IngestionStats:
    documents: int
    chunks: int


class IngestionPipeline:
    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        es_client: Elasticsearch,
        neo4j_driver: AsyncDriver,
        chroma_collection: str = "prod_docs",
        es_index: str = "prod_docs",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedder: BaseEmbedder | None = None,
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Cosine matches the unit-normalized SentenceTransformerEmbedder output; set
        # here too since whichever caller creates the collection first fixes its space.
        self._collection = chroma_client.get_or_create_collection(
            chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        self._es_client = es_client
        self._es_index = es_index
        self._graph_builder = graph_builder or GraphBuilder(neo4j_driver)
        self._ensure_es_index()

    async def run(self, loaders: Sequence[BaseLoader]) -> IngestionStats:
        documents: list[Document] = []
        for loader in loaders:
            documents.extend(loader.load())

        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunker.chunk(document))

        if not chunks:
            return IngestionStats(documents=len(documents), chunks=0)

        embeddings = self._embedder.embed([chunk.content for chunk in chunks])
        self._index_vector_store(chunks, embeddings)
        self._index_keyword_store(chunks)
        await self._graph_builder.build(chunks)
        return IngestionStats(documents=len(documents), chunks=len(chunks))

    def _ensure_es_index(self) -> None:
        if not self._es_client.indices.exists(index=self._es_index):
            self._es_client.indices.create(index=self._es_index, mappings=ES_INDEX_MAPPING)

    def _index_vector_store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # Upserting by chunk.id ("doc_id::chunk_index") makes re-ingestion idempotent.
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._to_chroma_metadata(chunk) for chunk in chunks],
        )

    def _index_keyword_store(self, chunks: list[Chunk]) -> None:
        # Bulk "index" actions keyed by chunk.id overwrite any prior version of the same chunk.
        actions = (
            {"_index": self._es_index, "_id": chunk.id, "_source": self._to_es_source(chunk)}
            for chunk in chunks
        )
        helpers.bulk(self._es_client, actions)

    @staticmethod
    def _to_chroma_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
        # Chroma metadata values must be flat scalars, so lists/None are normalized away.
        metadata: dict[str, str | int | float | bool] = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "source_type": chunk.source_type.value,
            "service_tags": ",".join(chunk.service_tags),
        }
        if chunk.url:
            metadata["url"] = chunk.url
        if chunk.severity:
            metadata["severity"] = chunk.severity
        if chunk.incident_date:
            metadata["incident_date"] = chunk.incident_date.isoformat()
        return metadata

    @staticmethod
    def _to_es_source(chunk: Chunk) -> dict:
        return {
            "content": chunk.content,
            "title": chunk.title,
            "tags": [*chunk.service_tags, chunk.source_type.value],
            "severity": chunk.severity,
            "service": chunk.service_tags,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "url": chunk.url,
            "source_type": chunk.source_type.value,
            "incident_date": chunk.incident_date.isoformat() if chunk.incident_date else None,
        }
