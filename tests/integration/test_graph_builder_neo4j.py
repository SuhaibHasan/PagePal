from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from ingestion.graph_builder import GraphBuilder
from ingestion.models import Chunk, SourceType

try:
    from testcontainers.community.neo4j import Neo4jContainer
except ImportError:  # pragma: no cover
    Neo4jContainer = None


class _FakeToolUseBlock:
    def __init__(self, input_data: dict) -> None:
        self.type = "tool_use"
        self.input = input_data


class _FakeMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class _ScriptedAnthropicMessages:
    """Returns a canned extraction payload keyed by a substring of the chunk text."""

    def __init__(self, payloads_by_content_marker: dict[str, dict]) -> None:
        self._payloads = payloads_by_content_marker

    async def create(self, *, messages, **kwargs):
        prompt_text = messages[0]["content"]
        for marker, payload in self._payloads.items():
            if marker in prompt_text:
                return _FakeMessage([_FakeToolUseBlock(payload)])
        return _FakeMessage([_FakeToolUseBlock({"entities": [], "relationships": []})])


class _ScriptedAnthropicClient:
    def __init__(self, payloads_by_content_marker: dict[str, dict]) -> None:
        self.messages = _ScriptedAnthropicMessages(payloads_by_content_marker)


def make_chunk(doc_id: str, chunk_index: int, content: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}::{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        content=content,
        title="Payment Outage Runbook",
        source_type=SourceType.PAGERDUTY,
    )


@pytest.fixture(scope="module")
def neo4j_container():
    if Neo4jContainer is None:
        pytest.skip("testcontainers[neo4j] is not installed")

    container = Neo4jContainer("neo4j:5.24-community")
    try:
        container.start()
    except Exception as exc:  # noqa: BLE001 - environment-dependent, not a code defect
        pytest.skip(f"Docker is unavailable to run the Neo4j test container: {exc}")

    yield container
    container.stop()


@pytest.fixture
async def driver(neo4j_container) -> AsyncIterator[AsyncDriver]:
    async_driver = AsyncGraphDatabase.driver(
        neo4j_container.get_connection_url(),
        auth=(neo4j_container.username, neo4j_container.password),
    )
    async with async_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield async_driver
    await async_driver.close()


async def _count(driver: AsyncDriver, query: str, **params) -> int:
    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()
        return record["count"]


async def test_build_upserts_typed_entities_and_relationship(driver: AsyncDriver):
    chunk = make_chunk(
        "pd-1",
        0,
        "The payment-service raised ERR-503 due to pool exhaustion. Escalated to the "
        "payments-team and resolved via the restart-runbook.",
    )
    payload = {
        "entities": [
            {"id": "e1", "name": "payment-service", "type": "SERVICE"},
            {"id": "e2", "name": "ERR-503", "type": "ERROR_CODE"},
            {"id": "e3", "name": "payments-team", "type": "TEAM"},
            {"id": "e4", "name": "restart-runbook", "type": "RUNBOOK"},
        ],
        "relationships": [
            {"from": "e2", "to": "e1", "type": "CAUSED_BY", "description": "pool exhaustion"},
            {"from": "e1", "to": "e3", "type": "OWNED_BY", "description": "escalation path"},
            {"from": "e2", "to": "e4", "type": "RESOLVED_BY", "description": "restart procedure"},
        ],
    }
    client = _ScriptedAnthropicClient({"payment-service": payload})
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([chunk])

    assert await _count(
        driver, "MATCH (n:SERVICE {name: $name}) RETURN count(n) AS count", name="payment-service"
    ) == 1
    assert await _count(
        driver, "MATCH (n:ERROR_CODE {name: $name}) RETURN count(n) AS count", name="ERR-503"
    ) == 1
    assert await _count(
        driver, "MATCH (n:TEAM {name: $name}) RETURN count(n) AS count", name="payments-team"
    ) == 1

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:ERROR_CODE {name: $err})-[r:CAUSED_BY]->(:SERVICE {name: $svc})
            RETURN r.description AS description, r.source_doc_id AS source_doc_id
            """,
            err="ERR-503",
            svc="payment-service",
        )
        record = await result.single()
        assert record["description"] == "pool exhaustion"
        assert record["source_doc_id"] == "pd-1"

        result = await session.run(
            """
            MATCH (:SERVICE {name: $svc})-[:MENTIONED_IN]->(c:Chunk {id: $chunk_id})
            RETURN c.doc_id AS doc_id
            """,
            svc="payment-service",
            chunk_id="pd-1::0",
        )
        record = await result.single()
        assert record["doc_id"] == "pd-1"


async def test_build_skips_relationship_referencing_unknown_entity_id(driver: AsyncDriver):
    chunk = make_chunk("pd-5", 0, "The payment-service is degraded.")
    payload = {
        "entities": [{"id": "e1", "name": "payment-service", "type": "SERVICE"}],
        # "e2" is never declared as an entity - this must not crash the build.
        "relationships": [
            {"from": "e2", "to": "e1", "type": "CAUSED_BY", "description": "unknown cause"}
        ],
    }
    client = _ScriptedAnthropicClient({"payment-service": payload})
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([chunk])

    assert await _count(
        driver, "MATCH (n:SERVICE {name: $name}) RETURN count(n) AS count", name="payment-service"
    ) == 1
    assert await _count(driver, "MATCH ()-[r:CAUSED_BY]->() RETURN count(r) AS count") == 0


async def test_build_is_idempotent_across_repeated_runs(driver: AsyncDriver):
    chunk = make_chunk("pd-2", 0, "The payment-service failed.")
    payload = {
        "entities": [{"id": "e1", "name": "payment-service", "type": "SERVICE"}],
        "relationships": [],
    }
    client = _ScriptedAnthropicClient({"payment-service": payload})
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([chunk])
    await builder.build([chunk])

    assert await _count(
        driver, "MATCH (n:SERVICE {name: $name}) RETURN count(n) AS count", name="payment-service"
    ) == 1
    assert await _count(
        driver, "MATCH (:SERVICE)-[m:MENTIONED_IN]->(:Chunk) RETURN count(m) AS count"
    ) == 1


async def test_build_dedupes_same_entity_mentioned_across_multiple_chunks(driver: AsyncDriver):
    chunk_a = make_chunk("pd-3", 0, "The payment-service is degraded.")
    chunk_b = make_chunk("pd-4", 0, "payment-service outage resolved.")
    payload = {
        "entities": [{"id": "e1", "name": "payment-service", "type": "SERVICE"}],
        "relationships": [],
    }
    client = _ScriptedAnthropicClient({"payment-service": payload})
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([chunk_a, chunk_b])

    assert await _count(
        driver, "MATCH (n:SERVICE {name: $name}) RETURN count(n) AS count", name="payment-service"
    ) == 1
    assert await _count(
        driver, "MATCH (:SERVICE)-[:MENTIONED_IN]->(c:Chunk) RETURN count(c) AS count"
    ) == 2
