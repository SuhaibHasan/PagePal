from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["vector", "keyword", "graph"]


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source_type: SourceType
    metadata: dict[str, Any] = Field(default_factory=dict)
