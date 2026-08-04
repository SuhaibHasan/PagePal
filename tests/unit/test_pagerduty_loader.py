import json
from pathlib import Path

from ingestion.loaders.pagerduty_loader import PagerDutyLoader
from ingestion.models import SourceType


def test_pagerduty_loader_parses_wrapped_export(tmp_path: Path):
    payload = {
        "incidents": [
            {
                "id": "INC123",
                "title": "Payment service outage",
                "urgency": "high",
                "created_at": "2024-06-01T12:00:00Z",
                "html_url": "https://example.pagerduty.com/incidents/INC123",
                "service": {"summary": "payment-service"},
                "log_entries": [{"summary": "Restarted pods"}],
            }
        ]
    }
    (tmp_path / "export.json").write_text(json.dumps(payload))

    documents = PagerDutyLoader(tmp_path).load()

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "pagerduty-INC123"
    assert document.source_type == SourceType.PAGERDUTY
    assert document.severity == "high"
    assert document.service_tags == ["payment-service"]
    assert "Restarted pods" in document.content
    assert document.incident_date is not None
    assert document.url == "https://example.pagerduty.com/incidents/INC123"


def test_pagerduty_loader_parses_bare_incident_object(tmp_path: Path):
    payload = {"id": "INC999", "title": "Redis latency spike"}
    (tmp_path / "single.json").write_text(json.dumps(payload))

    documents = PagerDutyLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].id == "pagerduty-INC999"
    assert documents[0].service_tags == []
