from __future__ import annotations

import re

from neo4j import Driver

from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


class Neo4jGraphRetriever(BaseRetriever):
    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        candidate_terms = {term.lower() for term in WORD_PATTERN.findall(query)}
        if not candidate_terms:
            return []

        with self._driver.session() as session:
            records = session.execute_read(self._find_related_chunks, candidate_terms, top_k)

        return [
            RetrievalResult(
                chunk_id=record["chunk_id"],
                document_id=record["document_id"],
                content=record["content"],
                score=float(record["hits"]),
                source_type="graph",
                metadata={"matched_entity": record["entity_name"]},
            )
            for record in records
        ]

    def ping(self) -> None:
        self._driver.verify_connectivity()

    @staticmethod
    def _find_related_chunks(tx, candidate_terms: set[str], top_k: int):
        result = tx.run(
            """
            MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)-[:PART_OF]->(d:Document)
            WHERE toLower(e.name) IN $terms
            RETURN c.id AS chunk_id, d.id AS document_id, c.content AS content,
                   e.name AS entity_name, count(*) AS hits
            ORDER BY hits DESC
            LIMIT $top_k
            """,
            terms=list(candidate_terms),
            top_k=top_k,
        )
        return list(result)
