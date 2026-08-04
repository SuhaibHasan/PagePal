from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    CONFLUENCE = "confluence"
    PAGERDUTY = "pagerduty"
    JIRA = "jira"


class Document(BaseModel):
    id: str
    title: str
    content: str
    source_type: SourceType
    url: str | None = None
    incident_date: datetime | None = None
    severity: str | None = None
    service_tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    chunk_index: int
    content: str
    title: str
    source_type: SourceType
    url: str | None = None
    incident_date: datetime | None = None
    severity: str | None = None
    service_tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
