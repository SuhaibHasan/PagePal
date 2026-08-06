from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from retrieval.answer_generator import Citation


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str
    services: dict[str, Any] = Field(default_factory=dict)
