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
uv run python -m ingestion \
  --local-dir path/to/runbooks \
  --pagerduty-dir path/to/pagerduty_exports \
  --confluence-space OPS \
  --jira-jql "project = OPS AND created >= -30d"
```

Pass any combination of the four flags; each corresponds to a loader:

| Flag                 | Source                                             |
|----------------------|-----------------------------------------------------|
| `--local-dir`        | Local `.md` / `.pdf` runbooks                       |
| `--pagerduty-dir`    | Directory of PagerDuty incident JSON exports        |
| `--confluence-space` | Confluence Cloud REST API (needs `CONFLUENCE_*` env)|
| `--jira-jql`         | Jira Cloud REST API search (needs `JIRA_*` env)     |

Credentials for Confluence/Jira are read from `CONFLUENCE_BASE_URL` / `CONFLUENCE_EMAIL` /
`CONFLUENCE_API_TOKEN` and `JIRA_BASE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` (see `.env.example`).

The pipeline chunks documents with a token-aware `RecursiveCharacterTextSplitter`
(512 tokens / 50 overlap), embeds chunks locally with `sentence-transformers`
(`all-MiniLM-L6-v2`, no external API), and upserts into:

- **ChromaDB** collection `prod_docs` — vector search
- **Elasticsearch** index `prod_docs` — BM25 over `content`, `title`, `tags`, `severity`, `service`
- **Neo4j** — `Document`/`Chunk`/`Entity` graph, built concurrently per document

Every chunk carries `title`, `url`, `source_type`, `incident_date`, `severity`, and
`service_tags` metadata through to all three stores. Chunk IDs are deterministic
(`doc_id::chunk_index`), and all three stores are upserted/merged by that ID, so re-running
the pipeline over the same sources is idempotent rather than creating duplicates.

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
