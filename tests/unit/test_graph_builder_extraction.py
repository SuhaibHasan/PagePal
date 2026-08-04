import anthropic
import httpx

from ingestion.graph_builder import EntityType, ExtractionResult, GraphBuilder
from ingestion.models import Chunk, SourceType


class _FakeToolUseBlock:
    def __init__(self, input_data: dict) -> None:
        self.type = "tool_use"
        self.input = input_data


class _FakeMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error=None) -> None:
        self.messages = _FakeMessages(response=response, error=error)


def make_chunk(content: str = "some incident text") -> Chunk:
    return Chunk(
        id="doc-1::0",
        doc_id="doc-1",
        chunk_index=0,
        content=content,
        title="Runbook",
        source_type=SourceType.MARKDOWN,
    )


async def test_extract_parses_tool_use_response():
    payload = {
        "entities": [{"id": "e1", "name": "payment-service", "type": "SERVICE"}],
        "relationships": [],
    }
    client = _FakeAnthropicClient(response=_FakeMessage([_FakeToolUseBlock(payload)]))
    builder = GraphBuilder(driver=object(), anthropic_client=client)

    result = await builder._extract(make_chunk())

    assert isinstance(result, ExtractionResult)
    assert result.entities[0].name == "payment-service"
    assert result.entities[0].type == EntityType.SERVICE

    sent_kwargs = client.messages.calls[0]
    assert sent_kwargs["tool_choice"] == {"type": "tool", "name": "record_graph_extraction"}
    assert "Extract entities and relationships" in sent_kwargs["messages"][0]["content"]


async def test_extract_parses_relationships_with_from_to_aliases():
    payload = {
        "entities": [
            {"id": "e1", "name": "payment-service", "type": "SERVICE"},
            {"id": "e2", "name": "ERR-503", "type": "ERROR_CODE"},
        ],
        "relationships": [
            {"from": "e2", "to": "e1", "type": "CAUSED_BY", "description": "pool exhaustion"}
        ],
    }
    client = _FakeAnthropicClient(response=_FakeMessage([_FakeToolUseBlock(payload)]))
    builder = GraphBuilder(driver=object(), anthropic_client=client)

    result = await builder._extract(make_chunk())

    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.from_id == "e2"
    assert relationship.to_id == "e1"
    assert relationship.description == "pool exhaustion"


async def test_extract_returns_empty_result_on_api_error():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    client = _FakeAnthropicClient(error=error)
    builder = GraphBuilder(driver=object(), anthropic_client=client)

    result = await builder._extract(make_chunk())

    assert result == ExtractionResult()


async def test_extract_returns_empty_result_on_invalid_entity_type():
    payload = {
        "entities": [{"id": "e1", "name": "x", "type": "NOT_A_REAL_TYPE"}],
        "relationships": [],
    }
    client = _FakeAnthropicClient(response=_FakeMessage([_FakeToolUseBlock(payload)]))
    builder = GraphBuilder(driver=object(), anthropic_client=client)

    result = await builder._extract(make_chunk())

    assert result == ExtractionResult()


async def test_extract_returns_empty_result_when_no_tool_use_block_present():
    client = _FakeAnthropicClient(response=_FakeMessage([]))
    builder = GraphBuilder(driver=object(), anthropic_client=client)

    result = await builder._extract(make_chunk())

    assert result == ExtractionResult()


async def test_build_skips_neo4j_write_when_no_entities_extracted():
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeToolUseBlock({"entities": [], "relationships": []})])
    )

    class _ExplodingDriver:
        def session(self):
            raise AssertionError("should not open a Neo4j session with no extracted entities")

    builder = GraphBuilder(driver=_ExplodingDriver(), anthropic_client=client)

    await builder.build([make_chunk()])
