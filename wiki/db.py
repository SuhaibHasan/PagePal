from __future__ import annotations

import asyncio

from elasticsearch import NotFoundError

from api.dependencies import get_elasticsearch_client
from wiki.schema import WikiEntry

WIKI_INDEX = "wiki_entries"

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
