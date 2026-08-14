from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from api.dependencies import get_async_anthropic_client
from wiki.db import get_wiki_entry, upsert_wiki_entry
from wiki.schema import WikiEntry

logger = logging.getLogger(__name__)

# Matches the model id used elsewhere in this codebase (ingestion/graph_builder.py,
# wiki/distiller.py, wiki/retriever.py) for Haiku extraction tasks.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_RESPONSE_TOKENS = 800

SYSTEM_PROMPT = (
    "You are a knowledge distiller. Given a question and its answer from a "
    "production support assistant, extract a wiki entry as JSON matching this "
    "schema: {title, summary, root_cause, resolution_steps, affected_services, "
    "related_error_codes, tags, severity_pattern, avg_resolution_time}. Only "
    "return JSON if the answer is specific and actionable. If the answer is "
    "vague, says it does not know, or lacks concrete resolution steps, return "
    "the string null."
)

NEW_ENTRY_CONFIDENCE = 0.6
CONFIDENCE_BUMP = 0.05
MAX_MERGED_CONFIDENCE = 0.95


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


async def upsert_from_rag(query: str, answer: str, citations: list[dict]) -> None:
    try:
        await _upsert_from_rag(query, answer, citations)
    except Exception:
        # Best-effort background enrichment - a distillation failure must never
        # surface as a chat-request failure.
        logger.exception("Wiki writer failed for query=%r", query)


async def _upsert_from_rag(query: str, answer: str, citations: list[dict]) -> None:
    client = get_async_anthropic_client()
    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=MAX_RESPONSE_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"}],
    )
    text_block = next((block for block in response.content if block.type == "text"), None)
    if text_block is None:
        logger.info("Wiki writer: no text block in response for query=%r", query)
        return

    text = text_block.text.strip()
    if text == "null":
        logger.info("Wiki writer: answer for query=%r was too vague to distill", query)
        return

    data = json.loads(text)
    now = datetime.now(UTC)
    # Built purely to get the deterministic title+tags id and the extracted content
    # fields; its confidence/timestamps are placeholders, overwritten below.
    candidate = WikiEntry(
        **data, confidence=0.0, created_at=now, last_updated=now, last_validated=now
    )

    existing = await get_wiki_entry(candidate.id)

    if existing is not None:
        entry = existing.model_copy(
            update={
                "resolution_steps": _dedupe_preserve_order(
                    existing.resolution_steps + candidate.resolution_steps
                ),
                "affected_services": _dedupe_preserve_order(
                    existing.affected_services + candidate.affected_services
                ),
                "related_error_codes": _dedupe_preserve_order(
                    existing.related_error_codes + candidate.related_error_codes
                ),
                "tags": _dedupe_preserve_order(existing.tags + candidate.tags),
                "confidence": min(existing.confidence + CONFIDENCE_BUMP, MAX_MERGED_CONFIDENCE),
                "last_updated": now,
            }
        )
    else:
        source_refs = _dedupe_preserve_order(
            [citation["title"] for citation in citations if citation.get("title")]
        )
        entry = candidate.model_copy(
            update={"confidence": NEW_ENTRY_CONFIDENCE, "source_refs": source_refs}
        )

    await upsert_wiki_entry(entry)
