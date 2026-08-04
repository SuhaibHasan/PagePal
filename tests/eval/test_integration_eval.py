import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="requires docker-compose services (chromadb, elasticsearch, neo4j); set RUN_INTEGRATION_TESTS=1",
)


def test_end_to_end_ingest_and_vector_retrieve(tmp_path: Path):
    import chromadb
    from elasticsearch import Elasticsearch
    from neo4j import GraphDatabase

    from ingestion.embedders.embedder import SentenceTransformerEmbedder
    from ingestion.loaders.file_loader import FileLoader
    from ingestion.pipeline import IngestionPipeline
    from retrieval.vector_retriever import ChromaVectorRetriever

    (tmp_path / "runbook.md").write_text(
        "The payment-service returns ERR-503 when the database connection pool is "
        "exhausted. Restart the payment-service pods to recover."
    )

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    es_client = Elasticsearch("http://localhost:9200")
    neo4j_driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "prodsupportbuddy")
    )

    collection_name = "eval_test_collection"
    index_name = "eval_test_index"
    embedder = SentenceTransformerEmbedder()
    pipeline = IngestionPipeline(
        chroma_client=chroma_client,
        es_client=es_client,
        neo4j_driver=neo4j_driver,
        chroma_collection=collection_name,
        es_index=index_name,
        embedder=embedder,
    )

    try:
        stats = pipeline.run([FileLoader(tmp_path)])
        assert stats.chunks > 0

        # Re-running against the same source must not create duplicate chunks.
        stats_again = pipeline.run([FileLoader(tmp_path)])
        assert stats_again.chunks == stats.chunks
        assert pipeline._collection.count() == stats.chunks

        retriever = ChromaVectorRetriever(chroma_client, embedder, collection_name)
        results = retriever.retrieve("why is payment-service returning errors", top_k=3)

        assert any("payment-service" in result.content for result in results)
    finally:
        chroma_client.delete_collection(collection_name)
        es_client.indices.delete(index=index_name, ignore_unavailable=True)
        neo4j_driver.close()
