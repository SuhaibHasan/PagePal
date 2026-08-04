from __future__ import annotations

from pathlib import Path

import chromadb
from elasticsearch import Elasticsearch, helpers
from neo4j import Driver

from ingestion.chunkers.text_chunker import TextChunker
from ingestion.embedders.embedder import DefaultEmbedder
from ingestion.graph_builder.builder import GraphBuilder
from ingestion.loaders.file_loader import FileLoader
from ingestion.models import Chunk


class IngestionPipeline:
    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        es_client: Elasticsearch,
        neo4j_driver: Driver,
        chroma_collection: str = "prod_support_chunks",
        es_index: str = "prod_support_chunks",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._embedder = DefaultEmbedder()
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._collection = chroma_client.get_or_create_collection(chroma_collection)
        self._es_client = es_client
        self._es_index = es_index
        self._graph_builder = GraphBuilder(neo4j_driver)

    def run(self, source_dir: str | Path) -> int:
        documents = FileLoader(source_dir).load()
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunker.chunk(document))

        if not chunks:
            return 0

        embeddings = self._embedder.embed([chunk.content for chunk in chunks])
        self._index_vector_store(chunks, embeddings)
        self._index_keyword_store(chunks)
        self._graph_builder.build(documents, chunks)
        return len(chunks)

    def _index_vector_store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[{**chunk.metadata, "document_id": chunk.document_id} for chunk in chunks],
        )

    def _index_keyword_store(self, chunks: list[Chunk]) -> None:
        actions = (
            {
                "_index": self._es_index,
                "_id": chunk.id,
                "_source": {
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "metadata": chunk.metadata,
                },
            }
            for chunk in chunks
        )
        helpers.bulk(self._es_client, actions)
