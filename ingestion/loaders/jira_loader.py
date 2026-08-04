from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from ingestion.loaders.base import BaseLoader
from ingestion.models import Document, SourceType


def _adf_to_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    parts = [node.get("text", "")] if node.get("type") == "text" else []
    for child in node.get("content") or []:
        text = _adf_to_text(child)
        if text:
            parts.append(text)
    return " ".join(part for part in parts if part)


class JiraLoader(BaseLoader):
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        jql: str = "order by created DESC",
        page_size: int = 50,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._jql = jql
        self._page_size = page_size
        self._client = client or httpx.Client(
            auth=(email, api_token),
            headers={"Accept": "application/json"},
            timeout=30.0,
        )

    def load(self) -> list[Document]:
        documents: list[Document] = []
        start_at = 0
        while True:
            response = self._client.get(
                f"{self._base_url}/rest/api/3/search",
                params={
                    "jql": self._jql,
                    "startAt": start_at,
                    "maxResults": self._page_size,
                    "fields": "summary,description,priority,labels,created",
                },
            )
            response.raise_for_status()
            payload = response.json()

            issues = payload.get("issues", [])
            documents.extend(self._to_document(issue) for issue in issues)

            start_at += len(issues)
            if not issues or start_at >= payload.get("total", 0):
                break

        return documents

    def _to_document(self, issue: dict[str, Any]) -> Document:
        fields = issue.get("fields", {})
        summary = fields.get("summary", issue["key"])
        description = _adf_to_text(fields.get("description") or {})

        created = fields.get("created")
        incident_date = datetime.fromisoformat(created) if created else None

        return Document(
            id=f"jira-{issue['key']}",
            title=summary,
            content=f"{summary}\n\n{description}".strip(),
            source_type=SourceType.JIRA,
            url=f"{self._base_url}/browse/{issue['key']}",
            incident_date=incident_date,
            severity=(fields.get("priority") or {}).get("name"),
            service_tags=fields.get("labels", []),
            extra={"issue_key": issue["key"]},
        )
