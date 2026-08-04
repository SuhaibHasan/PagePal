from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ingestion.loaders.base import BaseLoader
from ingestion.models import Document, SourceType


class PagerDutyLoader(BaseLoader):
    def __init__(self, export_dir: str | Path) -> None:
        self._export_dir = Path(export_dir)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self._export_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            incidents: list[dict[str, Any]]
            if isinstance(payload, dict):
                incidents = payload.get("incidents", [payload])
            else:
                incidents = payload
            documents.extend(self._to_document(incident) for incident in incidents)
        return documents

    def _to_document(self, incident: dict[str, Any]) -> Document:
        log_entries = incident.get("log_entries", [])
        log_text = "\n".join(entry.get("summary", "") for entry in log_entries if entry.get("summary"))
        content = "\n\n".join(
            part
            for part in (
                incident.get("title") or incident.get("summary", ""),
                incident.get("description", ""),
                log_text,
            )
            if part
        )

        incident_date = None
        created_at = incident.get("created_at")
        if created_at:
            incident_date = datetime.fromisoformat(created_at)

        service_name = incident.get("service", {}).get("summary")

        return Document(
            id=f"pagerduty-{incident['id']}",
            title=incident.get("title") or incident.get("summary", incident["id"]),
            content=content,
            source_type=SourceType.PAGERDUTY,
            url=incident.get("html_url"),
            incident_date=incident_date,
            severity=incident.get("urgency"),
            service_tags=[service_name] if service_name else [],
            extra={"status": incident.get("status", "")},
        )
