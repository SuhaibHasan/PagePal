from __future__ import annotations

import asyncio
import json
import logging

import anthropic
from elasticsearch import NotFoundError
from pydantic import BaseModel, Field, ValidationError

from api.dependencies import get_async_anthropic_client, get_elasticsearch_client
from wiki.db import WIKI_INDEX, increment_hit_count
from wiki.schema import WikiEntry

logger = logging.getLogger(__name__)

# Matches the model id used elsewhere in this codebase (ingestion/graph_builder.py,
# wiki/distiller.py) for Haiku extraction tasks.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_SIGNAL_TOKENS = 200

EXTRACTION_SYSTEM_PROMPT = (
    "Extract structured signals from this production support query. Return ONLY "
    "JSON: {services: [], error_codes: [], tags: [], intent: 'diagnosis|resolution|impact'}."
)

SERVICES_BOOST = 4.0
ERROR_CODES_BOOST = 3.0
TAGS_BOOST = 2.0
TEXT_BOOST = 1.0
# ES doesn't expose a natively-bounded relevance score, so "normalized" is defined
# here as raw _score divided by the maximum a hit could reach by satisfying every
# should clause at boost strength - i.e. the sum of the boosts above.
MAX_POSSIBLE_SCORE = SERVICES_BOOST + ERROR_CODES_BOOST + TAGS_BOOST + TEXT_BOOST

CONFIDENCE_FILTER_THRESHOLD = 0.5
NORMALIZED_SCORE_THRESHOLD = 0.75


def _normalize_score(raw_score: float) -> float:
    # Capped at 1.0: BM25 can over-score a hit past MAX_POSSIBLE_SCORE on
    # high-frequency terms, which would otherwise push "normalized" above 1.0.
    return min(raw_score / MAX_POSSIBLE_SCORE, 1.0)


class QuerySignals(BaseModel):
    services: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    intent: str | None = None


async def _extract_signals(query: str) -> QuerySignals:
    client = get_async_anthropic_client()
    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_SIGNAL_TOKENS,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            return QuerySignals()
        data = json.loads(text_block.text.strip())
        return QuerySignals.model_validate(data)
    except (anthropic.APIError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Wiki query signal extraction failed for %r: %s", query, exc)
        return QuerySignals()


def _build_query(query: str, signals: QuerySignals) -> dict:
    should: list[dict] = []
    if signals.services:
        should.append({"terms": {"affected_services": signals.services, "boost": SERVICES_BOOST}})
    if signals.error_codes:
        should.append(
            {"terms": {"related_error_codes": signals.error_codes, "boost": ERROR_CODES_BOOST}}
        )
    if signals.tags:
        should.append({"terms": {"tags": signals.tags, "boost": TAGS_BOOST}})
    should.append(
        {"multi_match": {"query": query, "fields": ["title", "summary"], "boost": TEXT_BOOST}}
    )

    return {
        "bool": {
            "should": should,
            "filter": [{"range": {"confidence": {"gte": CONFIDENCE_FILTER_THRESHOLD}}}],
            "minimum_should_match": 1,
        }
    }


# Holds references to fire-and-forget hit-count increments so they aren't
# garbage-collected before they run.
_background_tasks: set[asyncio.Task] = set()


async def retrieve(query: str) -> WikiEntry | None:
    signals = await _extract_signals(query)
    es_query = _build_query(query, signals)
    client = get_elasticsearch_client()

    try:
        response = await asyncio.to_thread(
            client.search, index=WIKI_INDEX, query=es_query, size=1
        )
    except NotFoundError:
        return None

    hits = response["hits"]["hits"]
    if not hits:
        return None

    hit = hits[0]
    entry = WikiEntry(id=hit["_id"], **hit["_source"])

    if entry.is_stale:
        return None

    normalized_score = _normalize_score(hit["_score"])
    if normalized_score < NORMALIZED_SCORE_THRESHOLD:
        return None

    task = asyncio.create_task(increment_hit_count(entry.id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return entry
