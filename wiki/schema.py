from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator


def _generate_id(title: str, tags: list[str]) -> str:
    payload = title + "|" + ",".join(sorted(tags))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WikiEntry(BaseModel):
    id: str = ""
    title: str
    summary: str
    root_cause: str | None = None
    resolution_steps: list[str] = Field(default_factory=list)
    affected_services: list[str] = Field(default_factory=list)
    related_error_codes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    severity_pattern: str | None = None
    avg_resolution_time: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    hit_count: int = 0
    source_refs: list[str] = Field(default_factory=list)
    # SHA256 of the concatenated source chunk content this entry was distilled
    # from - lets wiki/invalidator.py tell an unchanged source apart from an
    # updated one without redistilling on every staleness sweep. None for
    # entries predating this field.
    source_content_hash: str | None = None
    created_at: datetime
    last_updated: datetime
    last_validated: datetime
    ttl_days: int = 30

    @model_validator(mode="after")
    def _set_id_if_missing(self) -> WikiEntry:
        if not self.id:
            self.id = _generate_id(self.title, self.tags)
        return self

    @property
    def is_stale(self) -> bool:
        return (datetime.now(UTC) - self.last_updated).days > self.ttl_days
