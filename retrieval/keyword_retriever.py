from __future__ import annotations

import logging

import anthropic
from elasticsearch import Elasticsearch
from pydantic import BaseModel, ValidationError

from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

FILTER_EXTRACTION_PROMPT = (
    "Extract structured search filters from this production-incident query. "
    "Return JSON: { service, severity, error_code, date_range }. Use null for any field "
    "not mentioned in the query. date_range is { start, end } with dates in YYYY-MM-DD "
    "format, or null if no time period is mentioned."
)

FILTER_EXTRACTION_TOOL = {
    "name": "record_query_filters",
    "description": "Record structured filters extracted from a search query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {"type": ["string", "null"]},
            "severity": {"type": ["string", "null"]},
            "error_code": {"type": ["string", "null"]},
            "date_range": {
                "type": ["object", "null"],
                "properties": {
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                },
            },
        },
        "required": ["service", "severity", "error_code", "date_range"],
    },
}


class DateRange(BaseModel):
    start: str | None = None
    end: str | None = None


class QueryFilters(BaseModel):
    service: str | None = None
    severity: str | None = None
    error_code: str | None = None
    date_range: DateRange | None = None


class ElasticsearchKeywordRetriever(BaseRetriever):
    def __init__(
        self,
        client: Elasticsearch,
        index_name: str = "prod_docs",
        anthropic_client: anthropic.Anthropic | None = None,
        model: str = HAIKU_MODEL,
    ) -> None:
        self._client = client
        self._index_name = index_name
        self._llm = anthropic_client or anthropic.Anthropic()
        self._model = model

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, str] | None = None
    ) -> list[RetrievalResult]:
        extracted_filters = self._extract_filters(query)
        # Explicit filters from the caller (e.g. a UI dropdown) are more reliable
        # than the LLM's guess, so they win when both are present.
        if filters:
            if filters.get("service"):
                extracted_filters.service = filters["service"]
            if filters.get("severity"):
                extracted_filters.severity = filters["severity"]

        response = self._client.search(
            index=self._index_name,
            query=self._build_query(query, extracted_filters),
            size=top_k,
        )
        hits = response["hits"]["hits"]
        return [
            RetrievalResult(
                chunk_id=hit["_id"],
                document_id=hit["_source"].get("doc_id", hit["_id"]),
                content=hit["_source"].get("content", ""),
                score=hit["_score"],
                source_type="keyword",
                metadata={k: v for k, v in hit["_source"].items() if k != "content"},
            )
            for hit in hits
        ]

    def ping(self) -> bool:
        return bool(self._client.ping())

    def _extract_filters(self, query: str) -> QueryFilters:
        try:
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=512,
                tools=[FILTER_EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": FILTER_EXTRACTION_TOOL["name"]},
                messages=[
                    {
                        "role": "user",
                        "content": f"{FILTER_EXTRACTION_PROMPT}\n\nQuery: {query}",
                    }
                ],
            )
            tool_use = next(block for block in response.content if block.type == "tool_use")
            return QueryFilters.model_validate(tool_use.input)
        except (anthropic.APIError, ValidationError, StopIteration) as exc:
            logger.warning("Query filter extraction failed for %r: %s", query, exc)
            return QueryFilters()

    @staticmethod
    def _build_query(query: str, filters: QueryFilters) -> dict:
        must = [{"multi_match": {"query": query, "fields": ["content", "title", "tags", "service"]}}]
        filter_clauses: list[dict] = []

        # service/severity are indexed as keyword fields but casing varies by
        # source (PagerDuty urgency is lowercase, Jira priority is Title Case),
        # so match case-insensitively rather than requiring the LLM to guess casing.
        if filters.service:
            filter_clauses.append(
                {"term": {"service": {"value": filters.service, "case_insensitive": True}}}
            )
        if filters.severity:
            filter_clauses.append(
                {"term": {"severity": {"value": filters.severity, "case_insensitive": True}}}
            )
        if filters.error_code:
            # No dedicated error_code field is indexed, so filter on an exact
            # phrase match against content instead of a structured term filter.
            filter_clauses.append({"match_phrase": {"content": filters.error_code}})
        if filters.date_range and (filters.date_range.start or filters.date_range.end):
            range_clause = {}
            if filters.date_range.start:
                range_clause["gte"] = filters.date_range.start
            if filters.date_range.end:
                range_clause["lte"] = filters.date_range.end
            filter_clauses.append({"range": {"incident_date": range_clause}})

        return {"bool": {"must": must, "filter": filter_clauses}}
