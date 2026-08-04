from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from enum import Enum

import anthropic
from neo4j import AsyncDriver, AsyncManagedTransaction
from pydantic import BaseModel, Field, ValidationError

from ingestion.models import Chunk

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

EXTRACTION_PROMPT = (
    "Extract entities and relationships from this incident/runbook text. "
    "Return JSON: { entities: [{id, name, type}], relationships: [{from, to, type, description}] }. "
    "Entity types: SERVICE, ERROR_CODE, RUNBOOK, INCIDENT, TEAM, DEPENDENCY, CONFIG. "
    "Relationship types: CAUSED_BY, DEPENDS_ON, OWNED_BY, RESOLVED_BY, TRIGGERS, MITIGATED_BY."
)


class EntityType(str, Enum):
    SERVICE = "SERVICE"
    ERROR_CODE = "ERROR_CODE"
    RUNBOOK = "RUNBOOK"
    INCIDENT = "INCIDENT"
    TEAM = "TEAM"
    DEPENDENCY = "DEPENDENCY"
    CONFIG = "CONFIG"


class RelationshipType(str, Enum):
    CAUSED_BY = "CAUSED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    OWNED_BY = "OWNED_BY"
    RESOLVED_BY = "RESOLVED_BY"
    TRIGGERS = "TRIGGERS"
    MITIGATED_BY = "MITIGATED_BY"


class ExtractedEntity(BaseModel):
    id: str
    name: str
    type: EntityType


class ExtractedRelationship(BaseModel):
    model_config = {"populate_by_name": True}

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    type: RelationshipType
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


EXTRACTION_TOOL = {
    "name": "record_graph_extraction",
    "description": "Record the entities and relationships extracted from the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Local identifier used to reference this entity from relationships.",
                        },
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": [t.value for t in EntityType]},
                    },
                    "required": ["id", "name", "type"],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "type": {"type": "string", "enum": [t.value for t in RelationshipType]},
                        "description": {"type": "string"},
                    },
                    "required": ["from", "to", "type", "description"],
                },
            },
        },
        "required": ["entities", "relationships"],
    },
}


class GraphBuilder:
    def __init__(
        self,
        driver: AsyncDriver,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        model: str = HAIKU_MODEL,
        concurrency: int = 5,
    ) -> None:
        self._driver = driver
        self._llm = anthropic_client or anthropic.AsyncAnthropic()
        self._model = model
        self._semaphore = asyncio.Semaphore(concurrency)

    async def build(self, chunks: Sequence[Chunk]) -> None:
        await asyncio.gather(*(self._process_chunk(chunk) for chunk in chunks))

    async def _process_chunk(self, chunk: Chunk) -> None:
        async with self._semaphore:
            extraction = await self._extract(chunk)

        if not extraction.entities:
            return

        async with self._driver.session() as session:
            await session.execute_write(self._merge_extraction, chunk, extraction)

    async def _extract(self, chunk: Chunk) -> ExtractionResult:
        try:
            response = await self._llm.messages.create(
                model=self._model,
                max_tokens=2048,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": EXTRACTION_TOOL["name"]},
                messages=[
                    {
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}\n\nText:\n{chunk.content}",
                    }
                ],
            )
            tool_use = next(block for block in response.content if block.type == "tool_use")
            return ExtractionResult.model_validate(tool_use.input)
        except (anthropic.APIError, ValidationError, StopIteration) as exc:
            logger.warning("Graph extraction failed for chunk %s: %s", chunk.id, exc)
            return ExtractionResult()

    @staticmethod
    async def _merge_extraction(
        tx: AsyncManagedTransaction, chunk: Chunk, extraction: ExtractionResult
    ) -> None:
        entity_by_local_id = {entity.id: entity for entity in extraction.entities}

        for entity in extraction.entities:
            # Merging on (label, name) - rather than the LLM's per-call local id -
            # is what makes the same real-world entity dedupe across chunks/documents.
            await tx.run(
                f"""
                MERGE (e:{entity.type.value} {{name: $name}})
                MERGE (c:Chunk {{id: $chunk_id}})
                SET c.doc_id = $doc_id
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                name=entity.name,
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
            )

        for relationship in extraction.relationships:
            source = entity_by_local_id.get(relationship.from_id)
            target = entity_by_local_id.get(relationship.to_id)
            if source is None or target is None:
                logger.warning(
                    "Skipping relationship %s referencing unknown entity id in chunk %s",
                    relationship.type.value,
                    chunk.id,
                )
                continue

            await tx.run(
                f"""
                MATCH (a:{source.type.value} {{name: $from_name}})
                MATCH (b:{target.type.value} {{name: $to_name}})
                MERGE (a)-[r:{relationship.type.value}]->(b)
                SET r.description = $description, r.source_doc_id = $source_doc_id
                """,
                from_name=source.name,
                to_name=target.name,
                description=relationship.description,
                source_doc_id=chunk.doc_id,
            )
