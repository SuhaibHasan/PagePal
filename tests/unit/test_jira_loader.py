import httpx

from ingestion.loaders.jira_loader import JiraLoader, _adf_to_text
from ingestion.models import SourceType


def test_adf_to_text_flattens_nested_content():
    adf = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Database"},
                    {"type": "text", "text": "pool exhausted"},
                ],
            }
        ],
    }

    assert _adf_to_text(adf) == "Database pool exhausted"


def test_adf_to_text_handles_empty_input():
    assert _adf_to_text({}) == ""
    assert _adf_to_text(None) == ""


def test_jira_loader_maps_fields_and_flattens_description():
    issue = {
        "key": "OPS-42",
        "fields": {
            "summary": "Payment service 500s",
            "description": {
                "type": "doc",
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Spiking 500s"}]}
                ],
            },
            "priority": {"name": "High"},
            "labels": ["payment-service"],
            "created": "2024-06-01T12:00:00.000+0000",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issues": [issue], "total": 1})

    loader = JiraLoader(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    documents = loader.load()

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "jira-OPS-42"
    assert document.title == "Payment service 500s"
    assert "Spiking 500s" in document.content
    assert document.source_type == SourceType.JIRA
    assert document.severity == "High"
    assert document.service_tags == ["payment-service"]
    assert document.url == "https://example.atlassian.net/browse/OPS-42"
    assert document.incident_date is not None
