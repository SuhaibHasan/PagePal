import anthropic
import httpx

from retrieval.keyword_retriever import ElasticsearchKeywordRetriever


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


def _filters_client(payload: dict) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(response=_FakeMessage([_FakeToolUseBlock(payload)]))


NO_FILTERS_PAYLOAD = {"service": None, "severity": None, "error_code": None, "date_range": None}


class _FakeElasticsearch:
    def __init__(self, hits: list[dict] | None = None) -> None:
        self._hits = hits or []
        self.search_calls: list[dict] = []

    def search(self, *, index, query, size):
        self.search_calls.append({"index": index, "query": query, "size": size})
        return {"hits": {"hits": self._hits}}

    def ping(self) -> bool:
        return True


def make_hit(chunk_id: str = "doc-1::0", score: float = 5.4) -> dict:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "content": "The payment-service raised ERR-503.",
            "title": "Payment Outage",
            "doc_id": "doc-1",
            "service": ["payment-service"],
            "severity": "high",
        },
    }


def test_retrieve_builds_multi_match_across_expected_fields():
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(
        es, index_name="prod_docs", anthropic_client=_filters_client(NO_FILTERS_PAYLOAD)
    )

    retriever.retrieve("payment service errors", top_k=10)

    sent_query = es.search_calls[0]["query"]
    multi_match = sent_query["bool"]["must"][0]["multi_match"]
    assert multi_match["query"] == "payment service errors"
    assert set(multi_match["fields"]) == {"content", "title", "tags", "service"}
    assert sent_query["bool"]["filter"] == []
    assert es.search_calls[0]["size"] == 10


def test_retrieve_defaults_to_ten_results():
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(
        es, anthropic_client=_filters_client(NO_FILTERS_PAYLOAD)
    )

    retriever.retrieve("payment service errors")

    assert es.search_calls[0]["size"] == 10


def test_retrieve_maps_hits_to_results_with_bm25_score_and_metadata():
    es = _FakeElasticsearch([make_hit(chunk_id="doc-1::0", score=7.2)])
    retriever = ElasticsearchKeywordRetriever(
        es, anthropic_client=_filters_client(NO_FILTERS_PAYLOAD)
    )

    results = retriever.retrieve("payment service errors")

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "doc-1::0"
    assert result.document_id == "doc-1"
    assert result.score == 7.2
    assert result.source_type == "keyword"
    assert "content" not in result.metadata
    assert result.metadata["service"] == ["payment-service"]
    assert result.metadata["severity"] == "high"


def test_service_and_severity_extracted_as_case_insensitive_term_filters():
    payload = {"service": "payment-service", "severity": "High", "error_code": None, "date_range": None}
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(es, anthropic_client=_filters_client(payload))

    retriever.retrieve("payment service errors")

    filters = es.search_calls[0]["query"]["bool"]["filter"]
    assert {"term": {"service": {"value": "payment-service", "case_insensitive": True}}} in filters
    assert {"term": {"severity": {"value": "High", "case_insensitive": True}}} in filters


def test_error_code_extracted_as_phrase_filter_against_content():
    payload = {"service": None, "severity": None, "error_code": "ERR-503", "date_range": None}
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(es, anthropic_client=_filters_client(payload))

    retriever.retrieve("why did payment fail with ERR-503")

    filters = es.search_calls[0]["query"]["bool"]["filter"]
    assert {"match_phrase": {"content": "ERR-503"}} in filters


def test_date_range_extracted_as_range_filter_on_incident_date():
    payload = {
        "service": None,
        "severity": None,
        "error_code": None,
        "date_range": {"start": "2024-06-01", "end": "2024-06-30"},
    }
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(es, anthropic_client=_filters_client(payload))

    retriever.retrieve("incidents last June")

    filters = es.search_calls[0]["query"]["bool"]["filter"]
    assert {"range": {"incident_date": {"gte": "2024-06-01", "lte": "2024-06-30"}}} in filters


def test_partial_date_range_only_sets_provided_bound():
    payload = {
        "service": None,
        "severity": None,
        "error_code": None,
        "date_range": {"start": "2024-06-01", "end": None},
    }
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(es, anthropic_client=_filters_client(payload))

    retriever.retrieve("incidents since June")

    filters = es.search_calls[0]["query"]["bool"]["filter"]
    assert {"range": {"incident_date": {"gte": "2024-06-01"}}} in filters


def test_filter_extraction_failure_falls_back_to_unfiltered_search():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    es = _FakeElasticsearch([make_hit()])
    retriever = ElasticsearchKeywordRetriever(
        es, anthropic_client=_FakeAnthropicClient(error=error)
    )

    results = retriever.retrieve("payment service errors")

    assert es.search_calls[0]["query"]["bool"]["filter"] == []
    assert len(results) == 1


def test_ping_delegates_to_client():
    es = _FakeElasticsearch()
    retriever = ElasticsearchKeywordRetriever(
        es, anthropic_client=_filters_client(NO_FILTERS_PAYLOAD)
    )

    assert retriever.ping() is True
