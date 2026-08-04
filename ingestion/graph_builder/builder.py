from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import Driver

from ingestion.models import Chunk, Document

ERROR_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{2,6}\b")
SERVICE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,3}-service\b")


class GraphBuilder:
    def __init__(self, driver: Driver, max_workers: int = 8) -> None:
        self._driver = driver
        self._max_workers = max_workers

    def build(self, documents: list[Document], chunks: list[Chunk]) -> None:
        chunks_by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

        # Each document gets its own session so extraction runs concurrently;
        # the neo4j driver supports concurrent sessions from multiple threads.
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(self._build_for_document, document, chunks_by_doc.get(document.id, []))
                for document in documents
            ]
            for future in as_completed(futures):
                future.result()

    def _build_for_document(self, document: Document, chunks: list[Chunk]) -> None:
        with self._driver.session() as session:
            session.execute_write(self._merge_document, document, chunks)

    @staticmethod
    def _merge_document(tx, document: Document, chunks: list[Chunk]) -> None:
        tx.run(
            """
            MERGE (d:Document {id: $id})
            SET d.title = $title, d.url = $url, d.source_type = $source_type
            """,
            id=document.id,
            title=document.title,
            url=document.url,
            source_type=document.source_type.value,
        )
        for chunk in chunks:
            tx.run(
                """
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content, c.chunk_index = $chunk_index
                MERGE (c)-[:PART_OF]->(d)
                """,
                doc_id=document.id,
                chunk_id=chunk.id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
            )

            entities = set(ERROR_CODE_PATTERN.findall(chunk.content))
            entities |= set(SERVICE_PATTERN.findall(chunk.content))
            entities |= set(chunk.service_tags)
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
