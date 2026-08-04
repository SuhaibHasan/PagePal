# ProdSupportBuddy

A RAG-powered assistant that helps engineers resolve production incidents by retrieving
relevant runbooks, postmortems, and prior incidents from a hybrid vector/keyword/graph
knowledge base and answering with a Claude model.

## Structure

```
ingestion/    loaders, chunkers, embedders, and a Neo4j graph builder
retrieval/    vector (Chroma), keyword (Elasticsearch), and graph (Neo4j) retrievers + RRF reranker
api/          FastAPI service exposing /chat and /health
ui/           Next.js chat frontend
tests/        unit tests (ingestion/retrieval) and eval tests (retrieval metrics + integration)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Node.js 18+ for the UI
- Docker + Docker Compose for the backing stores

## Backing services

Start ChromaDB, Elasticsearch, Neo4j, and Redis:

```bash
docker compose up -d
```

| Service       | Port(s)      | Data volume     |
|---------------|--------------|-----------------|
| ChromaDB      | 8000         | chroma_data     |
| Elasticsearch | 9200         | es_data         |
| Neo4j         | 7474 / 7687  | neo4j_data      |
| Redis         | 6379         | redis_data      |

Neo4j Browser credentials default to `neo4j` / `prodsupportbuddy` (see `docker-compose.yml`).

## API

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
uv sync
uv run uvicorn api.main:app --reload --port 8080
```

## Ingest documents

```bash
uv run python -m ingestion path/to/runbooks
```

This loads `.md`/`.txt`/`.rst` files, chunks them, embeds and upserts chunks into ChromaDB,
indexes them in Elasticsearch, and builds Document/Chunk/Entity relationships in Neo4j.

## UI

```bash
cd ui
cp .env.local.example .env.local
npm install
npm run dev
```

## Tests

```bash
uv run pytest tests/unit             # fast, no external services
RUN_INTEGRATION_TESTS=1 uv run pytest tests/eval  # requires docker compose services running
```
