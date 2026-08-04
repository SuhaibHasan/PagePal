from __future__ import annotations

import re

from neo4j import Driver

from ingestion.models import Chunk, Document

ERROR_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{2,6}\b")
SERVICE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,3}-service\b")


class GraphBuilder:
    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def build(self, documents: list[Document], chunks: list[Chunk]) -> None:
        chunks_by_document: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

        with self._driver.session() as session:
            for document in documents:
                session.execute_write(
                    self._merge_document, document, chunks_by_document.get(document.id, [])
                )

    @staticmethod
    def _merge_document(tx, document: Document, chunks: list[Chunk]) -> None:
        tx.run(
            "MERGE (d:Document {id: $id}) SET d.title = $title, d.source = $source",
            id=document.id,
            title=document.title,
            source=document.source,
        )
        for chunk in chunks:
            tx.run(
                """
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content
                MERGE (c)-[:PART_OF]->(d)
                """,
                doc_id=document.id,
                chunk_id=chunk.id,
                content=chunk.content,
            )
            entities = set(ERROR_CODE_PATTERN.findall(chunk.content)) | set(
                SERVICE_PATTERN.findall(chunk.content)
            )
            for entity in entities:
                tx.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (e:Entity {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk.id,
                    name=entity,
                )
