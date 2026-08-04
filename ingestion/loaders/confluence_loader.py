from __future__ import annotations

import httpx

from ingestion.loaders.base import BaseLoader
from ingestion.loaders.html_utils import html_to_text
from ingestion.models import Document, SourceType


class ConfluenceLoader(BaseLoader):
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        space_key: str | None = None,
        page_size: int = 50,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._space_key = space_key
        self._page_size = page_size
        self._client = client or httpx.Client(
            auth=(email, api_token),
            headers={"Accept": "application/json"},
            timeout=30.0,
        )

    def load(self) -> list[Document]:
        documents: list[Document] = []
        start = 0
        while True:
            params: dict[str, str | int] = {
                "start": start,
                "limit": self._page_size,
                "expand": "body.storage,metadata.labels,space",
            }
            if self._space_key:
                params["spaceKey"] = self._space_key

            response = self._client.get(f"{self._base_url}/wiki/rest/api/content", params=params)
            response.raise_for_status()
            payload = response.json()

            results = payload.get("results", [])
            documents.extend(self._to_document(page) for page in results)

            if len(results) < self._page_size:
                break
            start += self._page_size

        return documents

    def _to_document(self, page: dict) -> Document:
        storage_html = page.get("body", {}).get("storage", {}).get("value", "")
        labels = [
            label["name"] for label in page.get("metadata", {}).get("labels", {}).get("results", [])
        ]
        webui_path = page.get("_links", {}).get("webui", "")

        return Document(
            id=f"confluence-{page['id']}",
            title=page.get("title", page["id"]),
            content=html_to_text(storage_html),
            source_type=SourceType.CONFLUENCE,
            url=f"{self._base_url}/wiki{webui_path}" if webui_path else None,
            service_tags=labels,
            extra={"space_key": page.get("space", {}).get("key", "")},
        )
