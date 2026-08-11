from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import anthropic
import tiktoken
from pydantic import ValidationError

from api.dependencies import get_async_anthropic_client
from wiki.db import upsert_wiki_entry
from wiki.schema import WikiEntry

logger = logging.getLogger(__name__)

# Matches the model id used elsewhere in this codebase (ingestion/graph_builder.py,
# retrieval/keyword_retriever.py) for Haiku extraction tasks.
HAIKU_MODEL = "claude-haiku-4-5-20251001"

MAX_CONTENT_TOKENS = 4000
MAX_RESPONSE_TOKENS = 1000
ENCODING_NAME = "cl100k_base"

SYSTEM_PROMPT = (
    "You are a knowledge distiller for a production support system. Given incident "
    "reports, runbooks, or support documentation, extract a structured wiki entry as "
    "JSON. Return ONLY valid JSON matching this schema exactly: {title, summary, "
    "root_cause, resolution_steps, affected_services, related_error_codes, tags, "
    "severity_pattern, avg_resolution_time}. If the content is too vague or generic "
    "to produce a useful wiki entry, return the string null."
)

RUNBOOK_CONFIDENCE = 0.85
DEFAULT_CONFIDENCE = 0.4
_CONFIDENCE_BY_SOURCE_AND_STATUS = {
    ("incident", "resolved"): 0.9,
    ("jira", "resolved"): 0.7,
    ("jira", "open"): 0.5,
}


def _concatenate_and_trim(chunks: list[dict]) -> str:
    # Trimming only shortens what's sent to the model - it never touches the chunk
    # dicts themselves, so metadata used later (source_refs, confidence) stays intact.
    content = "\n\n".join(chunk["content"] for chunk in chunks)
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(content)
    if len(tokens) > MAX_CONTENT_TOKENS:
        content = encoding.decode(tokens[:MAX_CONTENT_TOKENS])
    return content


def _confidence_for(chunks: list[dict]) -> float:
    metadata = chunks[0].get("metadata", {})
    source_type = metadata.get("source_type")
    if source_type == "runbook":
        return RUNBOOK_CONFIDENCE
    status = metadata.get("status")
    return _CONFIDENCE_BY_SOURCE_AND_STATUS.get((source_type, status), DEFAULT_CONFIDENCE)


async def distill(chunks: list[dict]) -> WikiEntry | None:
    if not chunks:
        return None

    content = _concatenate_and_trim(chunks)
    client = get_async_anthropic_client()

    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_RESPONSE_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        text_block = next((block for block in response.content if block.type == "text"), None)
    except anthropic.APIError as exc:
        logger.warning("Wiki distillation call failed: %s", exc)
        return None

    if text_block is None:
        logger.warning("Wiki distillation response contained no text block")
        return None
    text = text_block.text.strip()

    if text == "null":
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Wiki distillation returned invalid JSON: %s", exc)
        return None

    now = datetime.now(UTC)
    source_refs = list(dict.fromkeys(chunk["doc_id"] for chunk in chunks))

    try:
        entry = WikiEntry(
            **data,
            confidence=_confidence_for(chunks),
            source_refs=source_refs,
            created_at=now,
            last_updated=now,
            last_validated=now,
            hit_count=0,
            ttl_days=30,
        )
    except (TypeError, ValidationError) as exc:
        logger.warning("Wiki distillation returned a malformed entry: %s", exc)
        return None

    await upsert_wiki_entry(entry)
    return entry
