from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    content_snippet: str
    score: float
    source_type: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str
    services: dict[str, Any] = Field(default_factory=dict)
