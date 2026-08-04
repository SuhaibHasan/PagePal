from __future__ import annotations

import argparse
import os

import chromadb
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase

from ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ProdSupportBuddy stores")
    parser.add_argument("source_dir", help="Directory containing runbooks/postmortems to ingest")
    args = parser.parse_args()

    chroma_client = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", "8000")),
    )
    es_client = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    neo4j_driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "prodsupportbuddy"),
        ),
    )

    pipeline = IngestionPipeline(
        chroma_client=chroma_client,
        es_client=es_client,
        neo4j_driver=neo4j_driver,
        chroma_collection=os.environ.get("CHROMA_COLLECTION", "prod_support_chunks"),
        es_index=os.environ.get("ELASTICSEARCH_INDEX", "prod_support_chunks"),
    )
    chunk_count = pipeline.run(args.source_dir)
    print(f"Ingested {chunk_count} chunks from {args.source_dir}")


if __name__ == "__main__":
    main()
