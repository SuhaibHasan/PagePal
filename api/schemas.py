from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatFilters(BaseModel):
    service: str | None = None
    severity: str | None = None
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    filters: ChatFilters | None = None


class IngestRequest(BaseModel):
    source_type: Literal["local", "confluence", "pagerduty", "jira"]
    path: str | None = None  # local runbooks dir, or pagerduty export dir
    space_key: str | None = None  # confluence
    jql: str | None = None  # jira


class IngestResponse(BaseModel):
    status: str
    source_type: str


class GraphNode(BaseModel):
    id: str
    name: str
    labels: list[str]


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class SubgraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str
    services: dict[str, Any] = Field(default_factory=dict)
