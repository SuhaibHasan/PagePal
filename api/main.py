from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.config import Settings, get_settings
from api.dependencies import (
    get_answer_generator,
    get_graph_retriever,
    get_ingestion_pipeline,
    get_keyword_retriever,
    get_redis_client,
    get_reranker,
    get_vector_retriever,
)
from api.rag import RagPipeline
from api.rate_limit import is_within_rate_limit
from api.schemas import (
    ChatRequest,
    HealthStatus,
    IngestRequest,
    IngestResponse,
    SubgraphResponse,
    WikiSearchHit,
    WikiStats,
)
from api.telemetry import configure_tracing
from ingestion.loaders.base import BaseLoader
from ingestion.loaders.confluence_loader import ConfluenceLoader
from ingestion.loaders.file_loader import FileLoader
from ingestion.loaders.jira_loader import JiraLoader
from ingestion.loaders.pagerduty_loader import PagerDutyLoader
from ingestion.pipeline import IngestionPipeline
from retrieval.graph_retriever import Neo4jGraphRetriever
from retrieval.keyword_retriever import ElasticsearchKeywordRetriever
from retrieval.vector_retriever import ChromaVectorRetriever
from wiki.db import (
    create_wiki_index,
    get_wiki_entry,
    get_wiki_stats,
    search_wiki_entries,
    upsert_wiki_entry,
    wiki_index_exists,
)
from wiki.invalidator import invalidate_stale
from wiki.schema import WikiEntry

logger = logging.getLogger(__name__)

WIKI_INVALIDATION_INTERVAL_HOURS = 6
VALIDATION_CONFIDENCE_BUMP = 0.1
MAX_VALIDATED_CONFIDENCE = 0.95

configure_tracing()


def get_rag_pipeline(settings: Settings = Depends(get_settings)) -> RagPipeline:
    return RagPipeline(
        retrievers={
            "vector": get_vector_retriever(),
            "keyword": get_keyword_retriever(),
            "graph": get_graph_retriever(),
        },
        reranker=get_reranker(),
        answer_generator=get_answer_generator(),
        redis_client=get_redis_client(),
        settings=settings,
    )


def _build_loader(request: IngestRequest) -> BaseLoader:
    settings = get_settings()

    if request.source_type == "local":
        if not request.path:
            raise HTTPException(422, "path is required for source_type=local")
        return FileLoader(request.path)

    if request.source_type == "pagerduty":
        if not request.path:
            raise HTTPException(422, "path is required for source_type=pagerduty")
        return PagerDutyLoader(request.path)

    if request.source_type == "confluence":
        return ConfluenceLoader(
            base_url=settings.confluence_base_url,
            email=settings.confluence_email,
            api_token=settings.confluence_api_token,
            space_key=request.space_key,
        )

    return JiraLoader(
        base_url=settings.jira_base_url,
        email=settings.jira_email,
        api_token=settings.jira_api_token,
        jql=request.jql or "order by created DESC",
    )


async def _run_ingestion(pipeline: IngestionPipeline, loader: BaseLoader) -> None:
    try:
        stats = await pipeline.run([loader])
        logger.info(
            "Ingestion complete: %d chunks from %d documents", stats.chunks, stats.documents
        )
        if stats.chunks:
            logger.info("Wiki distillation tasks queued as part of ingestion")
    except Exception:
        logger.exception("Background ingestion failed")


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await create_wiki_index()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(invalidate_stale, "interval", hours=WIKI_INVALIDATION_INTERVAL_HOURS)
    scheduler.start()
    # Stashed on app.state so /health can report whether it's actually running -
    # it's otherwise only reachable from inside this context manager.
    app.state.wiki_scheduler = scheduler

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="ProdSupportBuddy API", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
async def health(
    request: Request,
    redis_client: redis.Redis = Depends(get_redis_client),
    vector_retriever: ChromaVectorRetriever = Depends(get_vector_retriever),
    keyword_retriever: ElasticsearchKeywordRetriever = Depends(get_keyword_retriever),
    graph_retriever: Neo4jGraphRetriever = Depends(get_graph_retriever),
) -> HealthStatus:
    services: dict[str, str] = {}

    for name, check in (
        ("redis", redis_client.ping),
        ("chromadb", vector_retriever.ping),
        ("elasticsearch", keyword_retriever.ping),
        ("neo4j", graph_retriever.ping),
    ):
        try:
            await asyncio.to_thread(check)
            services[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            services[name] = f"error: {exc}"

    try:
        services["wiki_index"] = (
            "ok" if await wiki_index_exists() else "error: wiki_entries index does not exist"
        )
    except Exception as exc:  # noqa: BLE001
        services["wiki_index"] = f"error: {exc}"

    scheduler = getattr(request.app.state, "wiki_scheduler", None)
    services["wiki_scheduler"] = (
        "ok" if scheduler is not None and scheduler.running else "error: scheduler not running"
    )

    overall = "ok" if all(status == "ok" for status in services.values()) else "degraded"
    return HealthStatus(status=overall, services=services)


@app.post("/chat")
async def chat(
    request: ChatRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> StreamingResponse:
    session_id = request.session_id or str(uuid.uuid4())

    if not is_within_rate_limit(redis_client, session_id):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded: 10 requests per minute per session"
        )

    filters = request.filters.model_dump(exclude_none=True) if request.filters else None

    async def event_stream():
        yield _format_sse("session", {"session_id": session_id})
        try:
            async for event, payload in pipeline.answer_stream(
                request.message, session_id, filters
            ):
                yield _format_sse(event, payload)
        except Exception as exc:
            logger.exception("Error while streaming chat response")
            yield _format_sse("error", {"error": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    loader = _build_loader(request)
    background_tasks.add_task(_run_ingestion, pipeline, loader)
    return IngestResponse(status="accepted", source_type=request.source_type)


@app.get("/graph/explore", response_model=SubgraphResponse)
def graph_explore(
    entity_name: str, graph_retriever: Neo4jGraphRetriever = Depends(get_graph_retriever)
) -> SubgraphResponse:
    return SubgraphResponse(**graph_retriever.explore_subgraph(entity_name))


wiki_router = APIRouter(prefix="/wiki")


# /search and /stats are registered before /{entry_id} so those literal paths
# aren't swallowed by the {entry_id} path parameter.
@wiki_router.get("/search", response_model=list[WikiSearchHit])
async def wiki_search(q: str, min_confidence: float = 0.5) -> list[WikiSearchHit]:
    results = await search_wiki_entries(q, min_confidence)
    return [WikiSearchHit(entry=entry, score=score) for entry, score in results]


@wiki_router.get("/stats", response_model=WikiStats)
async def wiki_stats() -> WikiStats:
    return WikiStats(**await get_wiki_stats())


@wiki_router.get("/{entry_id}", response_model=WikiEntry)
async def wiki_get(entry_id: str) -> WikiEntry:
    entry = await get_wiki_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Wiki entry not found")
    return entry


@wiki_router.post("/{entry_id}/validate", response_model=WikiEntry)
async def wiki_validate(entry_id: str, x_engineer_id: str = Header(...)) -> WikiEntry:
    entry = await get_wiki_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Wiki entry not found")

    logger.info("Wiki entry id=%s validated by engineer_id=%s", entry_id, x_engineer_id)

    validated = entry.model_copy(
        update={
            "confidence": min(entry.confidence + VALIDATION_CONFIDENCE_BUMP, MAX_VALIDATED_CONFIDENCE),
            "last_validated": datetime.now(UTC),
        }
    )
    await upsert_wiki_entry(validated)
    return validated


@wiki_router.delete("/{entry_id}", response_model=WikiEntry)
async def wiki_delete(entry_id: str) -> WikiEntry:
    entry = await get_wiki_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Wiki entry not found")

    # Soft delete: the document stays in ES with confidence zeroed out, rather
    # than being removed, so it drops out of search/stats without losing history.
    deleted = entry.model_copy(update={"confidence": 0.0, "last_updated": datetime.now(UTC)})
    await upsert_wiki_entry(deleted)
    return deleted


app.include_router(wiki_router)
