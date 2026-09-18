from __future__ import annotations

import argparse
import asyncio
import os

import chromadb
from elasticsearch import Elasticsearch
from neo4j import AsyncGraphDatabase

from ingestion.loaders.base import BaseLoader
from ingestion.loaders.confluence_loader import ConfluenceLoader
from ingestion.loaders.file_loader import FileLoader
from ingestion.loaders.jira_loader import JiraLoader
from ingestion.loaders.pagerduty_loader import PagerDutyLoader
from ingestion.pipeline import IngestionPipeline


def _build_loaders(args: argparse.Namespace) -> list[BaseLoader]:
    loaders: list[BaseLoader] = []

    if args.local_dir:
        loaders.append(FileLoader(args.local_dir))

    if args.pagerduty_dir:
        loaders.append(PagerDutyLoader(args.pagerduty_dir))

    if args.confluence_space:
        loaders.append(
            ConfluenceLoader(
                base_url=os.environ["CONFLUENCE_BASE_URL"],
                email=os.environ["CONFLUENCE_EMAIL"],
                api_token=os.environ["CONFLUENCE_API_TOKEN"],
                space_key=args.confluence_space,
            )
        )

    if args.jira_jql:
        loaders.append(
            JiraLoader(
                base_url=os.environ["JIRA_BASE_URL"],
                email=os.environ["JIRA_EMAIL"],
                api_token=os.environ["JIRA_API_TOKEN"],
                jql=args.jira_jql,
            )
        )

    return loaders


async def _run(loaders: list[BaseLoader]) -> None:
    chroma_client = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", "8000")),
    )
    es_client = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    neo4j_driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "prodsupportbuddy"),
        ),
    )

    try:
        pipeline = IngestionPipeline(
            chroma_client=chroma_client,
            es_client=es_client,
            neo4j_driver=neo4j_driver,
            chroma_collection=os.environ.get("CHROMA_COLLECTION", "prod_docs"),
            es_index=os.environ.get("ELASTICSEARCH_INDEX", "prod_docs"),
        )
        stats = await pipeline.run(loaders)
        print(f"Ingested {stats.chunks} chunks from {stats.documents} documents")
    finally:
        await neo4j_driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into PagePal stores")
    parser.add_argument("--local-dir", help="Directory of local markdown/PDF runbooks")
    parser.add_argument("--pagerduty-dir", help="Directory of PagerDuty incident JSON exports")
    parser.add_argument("--confluence-space", help="Confluence space key to ingest")
    parser.add_argument("--jira-jql", help="JQL query selecting Jira issues to ingest")
    args = parser.parse_args()

    loaders = _build_loaders(args)
    if not loaders:
        parser.error(
            "at least one of --local-dir, --pagerduty-dir, --confluence-space, "
            "--jira-jql is required"
        )

    asyncio.run(_run(loaders))


if __name__ == "__main__":
    main()
