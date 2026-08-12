from __future__ import annotations

import asyncio
from statistics import fmean

from elasticsearch import NotFoundError

from api.dependencies import get_elasticsearch_client
from wiki.schema import WikiEntry

WIKI_INDEX = "wiki_entries"

SEARCH_RESULT_SIZE = 5
# Pragmatic cap for a full-index scan; stats and staleness both need per-entry
# fields (is_stale) that ES can't aggregate without a scripted query, so this
# module fetches candidates and computes those in Python instead - fine at the
# scale a curated wiki index is expected to stay at.
MAX_SCAN_SIZE = 1000
HIGH_CONFIDENCE_THRESHOLD = 0.85

WIKI_INDEX_MAPPING = {
    "properties": {
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "summary": {"type": "text"},
        "tags": {"type": "keyword"},
        "affected_services": {"type": "keyword"},
        "related_error_codes": {"type": "keyword"},
        "confidence": {"type": "float"},
        "hit_count": {"type": "integer"},
        "last_updated": {"type": "date"},
        "last_validated": {"type": "date"},
        "created_at": {"type": "date"},
        "ttl_days": {"type": "integer"},
        "source_refs": {"type": "keyword"},
    }
}

INCREMENT_HIT_COUNT_SCRIPT = {"source": "ctx._source.hit_count += 1", "lang": "painless"}


async def create_wiki_index() -> None:
    # elasticsearch-py's client is synchronous, so blocking calls are offloaded to a
    # thread rather than blocking the event loop; the client instance itself (and its
    # underlying connection pool) is still the single one reused app-wide.
    client = get_elasticsearch_client()
    exists = await asyncio.to_thread(client.indices.exists, index=WIKI_INDEX)
    if not exists:
        await asyncio.to_thread(client.indices.create, index=WIKI_INDEX, mappings=WIKI_INDEX_MAPPING)


async def wiki_index_exists() -> bool:
    client = get_elasticsearch_client()
    return bool(await asyncio.to_thread(client.indices.exists, index=WIKI_INDEX))


async def upsert_wiki_entry(entry: WikiEntry) -> None:
    client = get_elasticsearch_client()
    doc = entry.model_dump(mode="json", exclude={"id"})
    await asyncio.to_thread(
        client.update, index=WIKI_INDEX, id=entry.id, doc=doc, doc_as_upsert=True
    )


async def get_wiki_entry(entry_id: str) -> WikiEntry | None:
    client = get_elasticsearch_client()
    try:
        response = await asyncio.to_thread(client.get, index=WIKI_INDEX, id=entry_id)
    except NotFoundError:
        return None
    return WikiEntry(id=response["_id"], **response["_source"])


async def increment_hit_count(entry_id: str) -> None:
    client = get_elasticsearch_client()
    await asyncio.to_thread(
        client.update, index=WIKI_INDEX, id=entry_id, script=INCREMENT_HIT_COUNT_SCRIPT
    )


async def search_wiki_entries(
    query: str, min_confidence: float, size: int = SEARCH_RESULT_SIZE
) -> list[tuple[WikiEntry, float]]:
    client = get_elasticsearch_client()
    es_query = {
        "bool": {
            "must": [{"multi_match": {"query": query, "fields": ["title", "summary", "tags"]}}],
            "filter": [{"range": {"confidence": {"gte": min_confidence}}}],
        }
    }
    try:
        response = await asyncio.to_thread(
            client.search, index=WIKI_INDEX, query=es_query, size=size
        )
    except NotFoundError:
        return []

    return [
        (WikiEntry(id=hit["_id"], **hit["_source"]), hit["_score"])
        for hit in response["hits"]["hits"]
    ]


async def get_wiki_stats() -> dict:
    client = get_elasticsearch_client()
    try:
        response = await asyncio.to_thread(
            client.search, index=WIKI_INDEX, query={"match_all": {}}, size=MAX_SCAN_SIZE
        )
        hits = response["hits"]["hits"]
    except NotFoundError:
        hits = []

    entries = [WikiEntry(id=hit["_id"], **hit["_source"]) for hit in hits]

    return {
        "total_entries": len(entries),
        "avg_confidence": fmean(entry.confidence for entry in entries) if entries else 0.0,
        "total_hits": sum(entry.hit_count for entry in entries),
        "stale_count": sum(1 for entry in entries if entry.is_stale),
        "high_confidence_count": sum(
            1 for entry in entries if entry.confidence >= HIGH_CONFIDENCE_THRESHOLD
        ),
    }
