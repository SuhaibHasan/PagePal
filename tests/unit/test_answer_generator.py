from retrieval.answer_generator import SYSTEM_PROMPT, AnswerGenerator
from retrieval.models import RetrievalResult


class _FakeStreamContext:
    def __init__(self, text_chunks: list[str]) -> None:
        self._text_chunks = text_chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        return iter(self._text_chunks)


class _FakeMessages:
    def __init__(self, text_chunks: list[str]) -> None:
        self._text_chunks = text_chunks
        self.stream_calls: list[dict] = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _FakeStreamContext(self._text_chunks)


class _FakeAnthropicClient:
    def __init__(self, answer_text: str) -> None:
        # split into a couple of chunks to exercise the streaming join path
        midpoint = max(1, len(answer_text) // 2)
        self.messages = _FakeMessages([answer_text[:midpoint], answer_text[midpoint:]])


def make_result(chunk_id: str, source_type: str, content: str = "chunk text", **metadata) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        content=content,
        score=1.0,
        source_type=source_type,
        metadata=metadata,
    )


def test_generate_labels_chunks_with_source_numbers_and_appends_question():
    client = _FakeAnthropicClient("no citations here")
    generator = AnswerGenerator(anthropic_client=client)
    context = [make_result("a::0", "vector", content="first chunk"), make_result("b::0", "keyword", content="second chunk")]

    generator.generate("why is payment failing", context)

    sent_message = client.messages.stream_calls[0]["messages"][0]["content"]
    assert "[Source 1]: first chunk" in sent_message
    assert "[Source 2]: second chunk" in sent_message
    assert "Question: why is payment failing" in sent_message


def test_generate_uses_system_prompt_and_configured_model():
    client = _FakeAnthropicClient("answer")
    generator = AnswerGenerator(anthropic_client=client, model="claude-sonnet-5")

    generator.generate("query", [make_result("a::0", "vector")])

    call = client.messages.stream_calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["model"] == "claude-sonnet-5"


def test_generate_includes_graph_context_section_when_paths_present():
    client = _FakeAnthropicClient("answer")
    generator = AnswerGenerator(anthropic_client=client)
    context = [
        make_result(
            "a::0",
            "graph",
            graph_paths=["auth-service → DEPENDS_ON → redis-cache"],
        )
    ]

    generator.generate("query", context)

    sent_message = client.messages.stream_calls[0]["messages"][0]["content"]
    assert "[Graph Context]:\nauth-service → DEPENDS_ON → redis-cache" in sent_message


def test_generate_omits_graph_context_section_when_no_paths():
    client = _FakeAnthropicClient("answer")
    generator = AnswerGenerator(anthropic_client=client)

    generator.generate("query", [make_result("a::0", "vector")])

    sent_message = client.messages.stream_calls[0]["messages"][0]["content"]
    assert "[Graph Context]" not in sent_message


def test_generate_limits_context_to_top_8_chunks():
    client = _FakeAnthropicClient("answer")
    generator = AnswerGenerator(anthropic_client=client)
    context = [make_result(f"c{i}::0", "vector", content=f"chunk {i}") for i in range(10)]

    generator.generate("query", context)

    sent_message = client.messages.stream_calls[0]["messages"][0]["content"]
    assert "[Source 8]:" in sent_message
    assert "[Source 9]:" not in sent_message


def test_generate_handles_empty_context():
    client = _FakeAnthropicClient("answer")
    generator = AnswerGenerator(anthropic_client=client)

    result = generator.generate("query", [])

    sent_message = client.messages.stream_calls[0]["messages"][0]["content"]
    assert "No relevant context was found" in sent_message
    assert result.citations == []


def test_generate_joins_streamed_text_into_final_answer():
    client = _FakeAnthropicClient("Restart the payment-service pods. [Source 1]")
    generator = AnswerGenerator(anthropic_client=client)

    result = generator.generate("query", [make_result("a::0", "vector", title="Runbook", url="https://x/a")])

    assert result.answer == "Restart the payment-service pods. [Source 1]"


def test_generate_maps_citations_back_to_title_url_and_retrieval_path():
    client = _FakeAnthropicClient("See [Source 1] and [Source 2].")
    generator = AnswerGenerator(anthropic_client=client)
    context = [
        make_result("a::0", "vector", title="Payment Runbook", url="https://x/a"),
        make_result("b::0", "keyword+vector", title="ES-503 Postmortem", url="https://x/b"),
    ]

    result = generator.generate("query", context)

    assert len(result.citations) == 2
    assert result.citations[0].title == "Payment Runbook"
    assert result.citations[0].url == "https://x/a"
    assert result.citations[0].retrieval_path == "vector"
    assert result.citations[0].graph_path is None
    assert result.citations[1].retrieval_path == "keyword+vector"


def test_generate_sets_graph_path_on_citation_when_source_is_graph():
    client = _FakeAnthropicClient("[Source 1] explains the cause.")
    generator = AnswerGenerator(anthropic_client=client)
    context = [
        make_result(
            "a::0",
            "graph",
            title="Redis Runbook",
            url="https://x/a",
            graph_paths=["auth-service → DEPENDS_ON → redis-cache → CAUSED_BY → INCIDENT-4521"],
        )
    ]

    result = generator.generate("query", context)

    assert result.citations[0].graph_path == "auth-service → DEPENDS_ON → redis-cache → CAUSED_BY → INCIDENT-4521"


def test_generate_ignores_out_of_range_citation_numbers():
    client = _FakeAnthropicClient("See [Source 1] and [Source 99].")
    generator = AnswerGenerator(anthropic_client=client)

    result = generator.generate("query", [make_result("a::0", "vector")])

    assert len(result.citations) == 1


def test_generate_dedupes_repeated_citation_numbers():
    client = _FakeAnthropicClient("[Source 1] ... also see [Source 1] again.")
    generator = AnswerGenerator(anthropic_client=client)

    result = generator.generate("query", [make_result("a::0", "vector")])

    assert len(result.citations) == 1


def test_generate_returns_no_citations_when_answer_cites_nothing():
    client = _FakeAnthropicClient("General guidance with no citations.")
    generator = AnswerGenerator(anthropic_client=client)

    result = generator.generate("query", [make_result("a::0", "vector")])

    assert result.citations == []
