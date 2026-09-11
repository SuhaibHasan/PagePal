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


class _RecordingSession:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, query, **params):
        self._calls.append(query)

    async def execute_write(self, func, *args):
        return await func(self, *args)


class _RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def session(self):
        return _RecordingSession(self.calls)


def _payload_for(entity_name: str) -> dict:
    return {
        "entities": [{"id": "e1", "name": entity_name, "type": "SERVICE"}],
        "relationships": [],
    }


async def test_build_creates_a_uniqueness_constraint_per_entity_label_before_writing():
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeToolUseBlock(_payload_for("payment-service"))])
    )
    driver = _RecordingDriver()
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([make_chunk()])

    constraint_calls = [call for call in driver.calls if "CONSTRAINT" in call]
    assert len(constraint_calls) == len(EntityType)
    assert all("IS UNIQUE" in call for call in constraint_calls)
    # The constraint(s) must exist before the entity MERGE that needs them for safety.
    merge_index = next(i for i, call in enumerate(driver.calls) if "MERGE" in call)
    assert merge_index > len(constraint_calls) - 1


async def test_build_only_creates_constraints_once_across_concurrent_chunks():
    # Regression test for a real race: build() processes chunks concurrently, and
    # without a uniqueness constraint, two chunks mentioning the same entity could
    # each MERGE-create their own node before either transaction committed.
    client = _FakeAnthropicClient(
        response=_FakeMessage([_FakeToolUseBlock(_payload_for("payment-service"))])
    )
    driver = _RecordingDriver()
    builder = GraphBuilder(driver=driver, anthropic_client=client)

    await builder.build([make_chunk("chunk a"), make_chunk("chunk b")])

    constraint_calls = [call for call in driver.calls if "CONSTRAINT" in call]
    assert len(constraint_calls) == len(EntityType)
