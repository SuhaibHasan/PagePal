import anthropic
import httpx

from retrieval.graph_retriever import Neo4jGraphRetriever


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

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error=None) -> None:
        self.messages = _FakeMessages(response=response, error=error)


def _entities_client(entities: list[str]) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(response=_FakeMessage([_FakeToolUseBlock({"entities": entities})]))


class _FakeNode:
    def __init__(self, element_id: str, name: str) -> None:
        self.element_id = element_id
        self._name = name

    def get(self, key, default=None):
        return self._name if key == "name" else default


class _FakeRelationship:
    def __init__(self, start_node: _FakeNode, end_node: _FakeNode, rel_type: str) -> None:
        self.start_node = start_node
        self.end_node = end_node
        self.type = rel_type


class _FakeSession:
    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = responses  # shared reference across sessions from the same driver
        self.queries: list[str] = []

    def execute_read(self, func, *args):
        return func(self, *args)

    def run(self, query, **params):
        self.queries.append(query)
        return self._responses.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeDriver:
    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = responses
        self.sessions: list[_FakeSession] = []

    def session(self):
        session = _FakeSession(self._responses)
        self.sessions.append(session)
        return session


class _FakeChromaCollection:
    def __init__(self, documents_by_id: dict[str, str]) -> None:
        self._documents_by_id = documents_by_id
        self.get_calls: list[list[str]] = []

    def get(self, ids):
        self.get_calls.append(ids)
        found = [(chunk_id, self._documents_by_id[chunk_id]) for chunk_id in ids if chunk_id in self._documents_by_id]
        return {
            "ids": [chunk_id for chunk_id, _ in found],
            "documents": [doc for _, doc in found],
        }


def test_retrieve_returns_empty_when_no_entities_extracted():
    driver = _FakeDriver(responses=[])
    retriever = Neo4jGraphRetriever(driver, anthropic_client=_entities_client([]))

    assert retriever.retrieve("what's the weather") == []
    assert driver.sessions == []


def test_retrieve_serializes_two_hop_path_and_hydrates_linked_chunk():
    auth = _FakeNode("n1", "auth-service")
    redis = _FakeNode("n2", "redis-cache")
    incident = _FakeNode("n3", "INCIDENT-4521")

    one_hop_row = {
        "n": auth,
        "r": [_FakeRelationship(auth, redis, "DEPENDS_ON")],
        "m": redis,
    }
    two_hop_row = {
        "n": auth,
        "r": [
            _FakeRelationship(auth, redis, "DEPENDS_ON"),
            _FakeRelationship(redis, incident, "CAUSED_BY"),
        ],
        "m": incident,
    }
    traversal_response = [one_hop_row, two_hop_row]
    chunk_lookup_response = [
        {"entity_name": "redis-cache", "chunk_id": "pd-1::0", "document_id": "pd-1"},
        {"entity_name": "INCIDENT-4521", "chunk_id": "pd-1::0", "document_id": "pd-1"},
    ]
    driver = _FakeDriver(responses=[traversal_response, chunk_lookup_response])
    chroma = _FakeChromaCollection({"pd-1::0": "Restart redis-cache to clear the connection pool."})
    retriever = Neo4jGraphRetriever(
        driver, chroma_collection=chroma, anthropic_client=_entities_client(["auth-service"])
    )

    results = retriever.retrieve("why is auth-service failing")

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "pd-1::0"
    assert result.document_id == "pd-1"
    assert result.source_type == "graph"
    assert "auth-service → DEPENDS_ON → redis-cache" in result.content
    assert "auth-service → DEPENDS_ON → redis-cache → CAUSED_BY → INCIDENT-4521" in result.content
    assert "Restart redis-cache" in result.content
    assert set(result.metadata["matched_entities"]) == {"redis-cache", "INCIDENT-4521"}
    assert result.score == 2.0


def test_retrieve_excludes_mentioned_in_hops_from_traversal_query():
    driver = _FakeDriver(responses=[[], []])
    retriever = Neo4jGraphRetriever(driver, anthropic_client=_entities_client(["payment-service"]))

    retriever.retrieve("payment-service issues")

    assert "MENTIONED_IN" in driver.sessions[0].queries[0]
    assert "r*1..2" in driver.sessions[0].queries[0]


def test_retrieve_returns_empty_when_traversal_finds_no_paths():
    driver = _FakeDriver(responses=[[]])
    retriever = Neo4jGraphRetriever(driver, anthropic_client=_entities_client(["payment-service"]))

    assert retriever.retrieve("payment-service issues") == []
    # only the traversal query should have run - no point looking up chunks for nothing
    assert len(driver.sessions) == 1


def test_retrieve_returns_path_only_content_when_chunk_not_in_chroma():
    auth = _FakeNode("n1", "auth-service")
    redis = _FakeNode("n2", "redis-cache")
    traversal_response = [{"n": auth, "r": [_FakeRelationship(auth, redis, "DEPENDS_ON")], "m": redis}]
    chunk_lookup_response = [{"entity_name": "redis-cache", "chunk_id": "pd-2::0", "document_id": "pd-2"}]
    driver = _FakeDriver(responses=[traversal_response, chunk_lookup_response])
    chroma = _FakeChromaCollection({})  # nothing hydrates
    retriever = Neo4jGraphRetriever(
        driver, chroma_collection=chroma, anthropic_client=_entities_client(["auth-service"])
    )

    results = retriever.retrieve("auth-service dependencies")

    assert results[0].content == "auth-service → DEPENDS_ON → redis-cache"


def test_entity_extraction_failure_returns_empty_results():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    driver = _FakeDriver(responses=[])
    retriever = Neo4jGraphRetriever(driver, anthropic_client=_FakeAnthropicClient(error=error))

    assert retriever.retrieve("payment-service issues") == []
    assert driver.sessions == []


def test_ping_verifies_connectivity():
    class _PingDriver:
        def __init__(self):
            self.pinged = False

        def verify_connectivity(self):
            self.pinged = True

    driver = _PingDriver()
    retriever = Neo4jGraphRetriever(driver, anthropic_client=_entities_client([]))

    retriever.ping()

    assert driver.pinged is True
