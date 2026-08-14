from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta

from api.config import get_settings
from api.dependencies import get_chroma_client, get_elasticsearch_client
from wiki.db import WIKI_INDEX, upsert_wiki_entry
from wiki.distiller import concatenate_and_trim, distill
from wiki.schema import WikiEntry

logger = logging.getLogger(__name__)

# WikiEntry.ttl_days is per-entry, but nothing in this codebase sets it away from
# the schema default - used as a cheap ES-side pre-filter. Each candidate is still
# re-checked against its own entry.is_stale before being processed, so a future
# entry with a non-default ttl_days is still handled correctly.
DEFAULT_TTL_DAYS = WikiEntry.model_fields["ttl_days"].default
MAX_CANDIDATES = 500

# Mirrors ingestion/pipeline.py's chunk -> distiller source_type mapping. Chroma
# chunk metadata only stores the raw SourceType.value (see
# ingestion/pipeline.py:_to_chroma_metadata) - the extraction-time "status" lives
# transiently in Document.extra and is never persisted to Chroma or Elasticsearch,
# so entries rebuilt here can't recover the original resolved/open distinction and
# fall back to the distiller's default confidence bucket for non-runbook sources.
_DISTILLER_SOURCE_TYPE_BY_SOURCE_TYPE = {
    "pagerduty": "incident",
    "jira": "jira",
}


def _to_distiller_chunk(doc_id: str, content: str, metadata: dict) -> dict:
    raw_source_type = metadata.get("source_type", "")
    source_type = _DISTILLER_SOURCE_TYPE_BY_SOURCE_TYPE.get(raw_source_type, "runbook")
    return {"doc_id": doc_id, "content": content, "metadata": {"source_type": source_type, "status": None}}


def _get_collection():
    settings = get_settings()
    return get_chroma_client().get_or_create_collection(settings.chroma_collection)


async def _fetch_chunks_for_doc(collection, doc_id: str) -> list[dict]:
    result = await asyncio.to_thread(
        collection.get, where={"doc_id": doc_id}, include=["documents", "metadatas"]
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        _to_distiller_chunk(doc_id, document, metadata)
        for document, metadata in zip(documents, metadatas, strict=True)
    ]


async def _fetch_chunks_for_docs(collection, doc_ids: list[str]) -> list[dict]:
    chunks: list[dict] = []
    for doc_id in doc_ids:
        chunks.extend(await _fetch_chunks_for_doc(collection, doc_id))
    return chunks


async def _process_stale_entry(entry: WikiEntry, collection, counts: dict[str, int]) -> None:
    chunks = await _fetch_chunks_for_docs(collection, entry.source_refs)

    if not chunks:
        soft_deleted = entry.model_copy(update={"confidence": 0.0})
        await upsert_wiki_entry(soft_deleted)
        counts["soft_deleted"] += 1
        return

    current_hash = hashlib.sha256(concatenate_and_trim(chunks).encode()).hexdigest()
    if entry.source_content_hash is not None and current_hash == entry.source_content_hash:
        refreshed = entry.model_copy(update={"last_validated": datetime.now(UTC)})
        await upsert_wiki_entry(refreshed)
        counts["refreshed"] += 1
        return

    await distill(chunks)
    counts["redistilled"] += 1


async def invalidate_stale() -> dict[str, int]:
    counts = {"refreshed": 0, "redistilled": 0, "soft_deleted": 0}

    try:
        es_client = get_elasticsearch_client()
        collection = _get_collection()

        threshold = datetime.now(UTC) - timedelta(days=DEFAULT_TTL_DAYS)
        query = {"range": {"last_updated": {"lt": threshold.isoformat()}}}
        response = await asyncio.to_thread(
            es_client.search, index=WIKI_INDEX, query=query, size=MAX_CANDIDATES
        )
    except Exception:
        # Runs on a 6-hour APScheduler interval - a failed sweep should just be
        # retried next cycle, not surface anywhere a request is waiting on it.
        logger.exception("Wiki staleness sweep failed to query Elasticsearch")
        return counts

    for hit in response["hits"]["hits"]:
        entry = WikiEntry(id=hit["_id"], **hit["_source"])
        if not entry.is_stale:
            continue

        try:
            await _process_stale_entry(entry, collection, counts)
        except Exception:
            logger.exception("Failed to process stale wiki entry id=%s", entry.id)

    return counts


async def invalidate_for_doc(doc_id: str) -> None:
    try:
        es_client = get_elasticsearch_client()
        response = await asyncio.to_thread(
            es_client.search,
            index=WIKI_INDEX,
            query={"term": {"source_refs": doc_id}},
            size=MAX_CANDIDATES,
        )
        hits = response["hits"]["hits"]

        if hits:
            chunks = await _fetch_chunks_for_doc(_get_collection(), doc_id)
            if chunks:
                await distill(chunks)

        logger.info("Wiki invalidated for doc_id=%s, entries=%d", doc_id, len(hits))
    except Exception:
        logger.exception("Wiki invalidation failed for doc_id=%s", doc_id)
