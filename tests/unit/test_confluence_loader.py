import httpx

from ingestion.loaders.confluence_loader import ConfluenceLoader
from ingestion.models import SourceType


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_confluence_loader_parses_storage_html_and_labels():
    page = {
        "id": "123",
        "title": "Payment Outage Runbook",
        "body": {"storage": {"value": "<p>Restart the <b>payment-service</b>.</p>"}},
        "metadata": {"labels": {"results": [{"name": "payments"}, {"name": "runbook"}]}},
        "space": {"key": "OPS"},
        "_links": {"webui": "/spaces/OPS/pages/123"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [page], "size": 1})

    loader = ConfluenceLoader(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="token",
        client=_client(handler),
    )

    documents = loader.load()

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "confluence-123"
    assert document.title == "Payment Outage Runbook"
    assert "Restart the" in document.content
    assert "payment-service" in document.content
    assert document.source_type == SourceType.CONFLUENCE
    assert document.url == "https://example.atlassian.net/wiki/spaces/OPS/pages/123"
    assert set(document.service_tags) == {"payments", "runbook"}
    assert document.extra["space_key"] == "OPS"


def test_confluence_loader_paginates_through_multiple_pages():
    empty_page = {"body": {}, "metadata": {}, "space": {}, "_links": {}}
    page_1 = [{**empty_page, "id": "1", "title": "Page 1"}, {**empty_page, "id": "2", "title": "Page 2"}]
    page_2 = [{**empty_page, "id": "3", "title": "Page 3"}]
    responses = [page_1, page_2]

    def handler(request: httpx.Request) -> httpx.Response:
        pages = responses.pop(0)
        return httpx.Response(200, json={"results": pages, "size": len(pages)})

    loader = ConfluenceLoader(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="token",
        page_size=2,
        client=_client(handler),
    )

    documents = loader.load()

    assert [document.id for document in documents] == ["confluence-1", "confluence-2", "confluence-3"]
