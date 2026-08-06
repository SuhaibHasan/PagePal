from __future__ import annotations

import json
import logging
import uuid

import redis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
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
from api.schemas import ChatRequest, HealthStatus, IngestRequest, IngestResponse, SubgraphResponse
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

logger = logging.getLogger(__name__)

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
    except Exception:
        logger.exception("Background ingestion failed")


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


app = FastAPI(title="ProdSupportBuddy API", version="0.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
def health(
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
            check()
            services[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            services[name] = f"error: {exc}"

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
