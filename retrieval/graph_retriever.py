from __future__ import annotations

import re

import chromadb
from neo4j import Driver

from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


class Neo4jGraphRetriever(BaseRetriever):
    def __init__(self, driver: Driver, chroma_collection: chromadb.Collection | None = None) -> None:
        self._driver = driver
        # Neo4j only stores the chunk_id join key; actual chunk text is hydrated
        # from Chroma so content never has to be duplicated across stores.
        self._chroma_collection = chroma_collection

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        candidate_terms = {term.lower() for term in WORD_PATTERN.findall(query)}
        if not candidate_terms:
            return []

        with self._driver.session() as session:
            records = session.execute_read(self._find_related_chunks, candidate_terms, top_k)

        if not records:
            return []

        contents = self._hydrate_contents([record["chunk_id"] for record in records])

        return [
            RetrievalResult(
                chunk_id=record["chunk_id"],
                document_id=record["document_id"] or "",
                content=contents.get(record["chunk_id"], ""),
                score=float(record["hits"]),
                source_type="graph",
                metadata={"matched_entity": record["entity_name"]},
            )
            for record in records
        ]

    def ping(self) -> None:
        self._driver.verify_connectivity()

    def _hydrate_contents(self, chunk_ids: list[str]) -> dict[str, str]:
        if self._chroma_collection is None or not chunk_ids:
            return {}
        result = self._chroma_collection.get(ids=chunk_ids)
        return dict(zip(result.get("ids", []), result.get("documents", []), strict=False))

    @staticmethod
    def _find_related_chunks(tx, candidate_terms: set[str], top_k: int):
        result = tx.run(
            """
            MATCH (e) WHERE toLower(e.name) IN $terms
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.id AS chunk_id, c.doc_id AS document_id,
                   e.name AS entity_name, count(*) AS hits
            ORDER BY hits DESC
            LIMIT $top_k
            """,
            terms=list(candidate_terms),
            top_k=top_k,
        )
        return list(result)
