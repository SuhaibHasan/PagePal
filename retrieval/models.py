from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Individual retrievers tag their own results "vector" / "keyword" / "graph".
# After reranking dedupes across retrievers, a result may carry a combined tag
# like "keyword+vector" - so this is a plain str rather than a fixed Literal.
SourceType = str


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source_type: SourceType
    metadata: dict[str, Any] = Field(default_factory=dict)
