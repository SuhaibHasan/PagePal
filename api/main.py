from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings, get_settings
from api.dependencies import (
    get_answer_generator,
    get_graph_retriever,
    get_keyword_retriever,
    get_redis_client,
    get_reranker,
    get_vector_retriever,
)
from api.rag import RagPipeline
from api.schemas import ChatRequest, ChatResponse, HealthStatus


def get_rag_pipeline(settings: Settings = Depends(get_settings)) -> RagPipeline:
    return RagPipeline(
        retrievers=[get_vector_retriever(), get_keyword_retriever(), get_graph_retriever()],
        reranker=get_reranker(),
        answer_generator=get_answer_generator(),
        redis_client=get_redis_client(),
        settings=settings,
    )


app = FastAPI(title="ProdSupportBuddy API", version="0.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    services: dict[str, str] = {}

    for name, check in (
        ("redis", lambda: get_redis_client().ping()),
        ("chromadb", lambda: get_vector_retriever().ping()),
        ("elasticsearch", lambda: get_keyword_retriever().ping()),
        ("neo4j", lambda: get_graph_retriever().ping()),
    ):
        try:
            check()
            services[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            services[name] = f"error: {exc}"

    overall = "ok" if all(status == "ok" for status in services.values()) else "degraded"
    return HealthStatus(status=overall, services=services)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline: RagPipeline = Depends(get_rag_pipeline)) -> ChatResponse:
    return await pipeline.answer(request.message)
