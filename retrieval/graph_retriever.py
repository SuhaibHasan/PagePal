from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import anthropic
import chromadb
from neo4j import Driver
from pydantic import BaseModel, Field, ValidationError

from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

ENTITY_EXTRACTION_PROMPT = (
    "Extract the specific services, error codes, and incident identifiers mentioned or "
    'implied by this production-support query. Return JSON: { entities: [name, ...] }. '
    'Only include concrete identifiers (e.g. "payment-service", "ERR-503", '
    '"INCIDENT-4521"), not generic words.'
)

ENTITY_EXTRACTION_TOOL = {
    "name": "record_query_entities",
    "description": "Record the entity names mentioned in a search query.",
    "input_schema": {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
        "required": ["entities"],
    },
}


class QueryEntities(BaseModel):
    entities: list[str] = Field(default_factory=list)


@dataclass
class _GraphPath:
    text: str
    node_names: list[str]


@dataclass
class _ChunkPaths:
    document_id: str
    entity_names: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)


class Neo4jGraphRetriever(BaseRetriever):
    def __init__(
        self,
        driver: Driver,
        chroma_collection: chromadb.Collection | None = None,
        anthropic_client: anthropic.Anthropic | None = None,
        model: str = HAIKU_MODEL,
    ) -> None:
        self._driver = driver
        # Neo4j only stores the chunk_id join key; actual chunk text is hydrated
        # from Chroma so content never has to be duplicated across stores.
        self._chroma_collection = chroma_collection
        self._llm = anthropic_client or anthropic.Anthropic()
        self._model = model

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        entities = self._extract_entities(query)
        if not entities:
            return []

        with self._driver.session() as session:
            path_rows = session.execute_read(self._traverse_subgraph, entities, top_k * 5)

        paths = self._serialize_paths(path_rows)
        if not paths:
            return []

        node_names = {name for path in paths for name in path.node_names}
        with self._driver.session() as session:
            chunk_rows = session.execute_read(self._find_chunk_ids, list(node_names))

        chunks_by_id = self._group_paths_by_chunk(paths, chunk_rows)
        if not chunks_by_id:
            return []

        return self._to_results(chunks_by_id, top_k)

    def ping(self) -> None:
        self._driver.verify_connectivity()

    def _extract_entities(self, query: str) -> list[str]:
        try:
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=512,
                tools=[ENTITY_EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": ENTITY_EXTRACTION_TOOL["name"]},
                messages=[
                    {"role": "user", "content": f"{ENTITY_EXTRACTION_PROMPT}\n\nQuery: {query}"}
                ],
            )
            tool_use = next(block for block in response.content if block.type == "tool_use")
            return QueryEntities.model_validate(tool_use.input).entities
        except (anthropic.APIError, ValidationError, StopIteration) as exc:
            logger.warning("Entity extraction failed for %r: %s", query, exc)
            return []

    def _hydrate_contents(self, chunk_ids: list[str]) -> dict[str, str]:
        if self._chroma_collection is None or not chunk_ids:
            return {}
        result = self._chroma_collection.get(ids=chunk_ids)
        return dict(zip(result.get("ids", []), result.get("documents", []), strict=False))

    def _to_results(self, chunks_by_id: dict[str, _ChunkPaths], top_k: int) -> list[RetrievalResult]:
        contents = self._hydrate_contents(list(chunks_by_id.keys()))
        ranked = sorted(chunks_by_id.items(), key=lambda item: len(item[1].paths), reverse=True)

        results = []
        for chunk_id, info in ranked[:top_k]:
            path_text = "\n".join(sorted(info.paths))
            chunk_text = contents.get(chunk_id, "")
            content = f"{path_text}\n\n{chunk_text}".strip() if chunk_text else path_text
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=info.document_id,
                    content=content,
                    score=float(len(info.paths)),
                    source_type="graph",
                    metadata={
                        "matched_entities": sorted(info.entity_names),
                        "graph_paths": sorted(info.paths),
                    },
                )
            )
        return results

    @staticmethod
    def _serialize_paths(path_rows) -> list[_GraphPath]:
        seen_texts: set[str] = set()
        paths: list[_GraphPath] = []
        for record in path_rows:
            start_node = record["n"]
            current = start_node
            nodes = [start_node]
            parts = [start_node.get("name", "?")]
            for rel in record["r"]:
                next_node = rel.end_node if rel.start_node.element_id == current.element_id else rel.start_node
                parts.append(rel.type)
                parts.append(next_node.get("name", "?"))
                nodes.append(next_node)
                current = next_node

            text = " → ".join(parts)
            if text in seen_texts:
                continue
            seen_texts.add(text)
            paths.append(_GraphPath(text=text, node_names=[node.get("name", "?") for node in nodes]))
        return paths

    @staticmethod
    def _group_paths_by_chunk(paths: list[_GraphPath], chunk_rows) -> dict[str, _ChunkPaths]:
        chunks_by_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in chunk_rows:
            chunks_by_entity[row["entity_name"]].append((row["chunk_id"], row["document_id"]))

        grouped: dict[str, _ChunkPaths] = {}
        for path in paths:
            for node_name in path.node_names:
                for chunk_id, document_id in chunks_by_entity.get(node_name, []):
                    entry = grouped.setdefault(chunk_id, _ChunkPaths(document_id=document_id))
                    entry.entity_names.add(node_name)
                    entry.paths.add(path.text)
        return grouped

    @staticmethod
    def _traverse_subgraph(tx, entities: list[str], limit: int):
        # MENTIONED_IN is excluded so traversal stays within the semantic entity
        # graph (SERVICE/ERROR_CODE/RUNBOOK/...) instead of hopping through the
        # bipartite entity<->Chunk linkage edges, which would produce nonsense
        # paths through unrelated entities that merely share a source chunk.
        result = tx.run(
            """
            MATCH (n)-[r*1..2]-(m)
            WHERE toLower(n.name) IN $entities
              AND ALL(rel IN r WHERE type(rel) <> 'MENTIONED_IN')
            RETURN n, r, m
            LIMIT $limit
            """,
            entities=[name.lower() for name in entities],
            limit=limit,
        )
        return list(result)

    @staticmethod
    def _find_chunk_ids(tx, node_names: list[str]):
        result = tx.run(
            """
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.name IN $names
            RETURN e.name AS entity_name, c.id AS chunk_id, c.doc_id AS document_id
            """,
            names=node_names,
        )
        return list(result)
