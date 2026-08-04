# ProdSupportBuddy

A RAG-powered assistant that helps engineers resolve production incidents by retrieving
relevant runbooks, postmortems, and prior incidents from a hybrid vector/keyword/graph
knowledge base and answering with a Claude model.

## Structure

```
ingestion/    loaders, chunkers, embedders, and an LLM-based Neo4j graph builder
retrieval/    vector (Chroma), keyword (Elasticsearch), and graph (Neo4j) retrievers + RRF reranker
api/          FastAPI service exposing /chat and /health
ui/           Next.js chat frontend
tests/        unit tests, eval tests (retrieval metrics + full-stack integration),
              and a Neo4j-testcontainer suite for the graph builder
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

Every chunk carries `title`, `url`, `source_type`, `incident_date`, `severity`, and
`service_tags` metadata through to both stores. Chunk IDs are deterministic
(`doc_id::chunk_index`), and both stores are upserted by that ID, so re-running the
pipeline over the same sources is idempotent rather than creating duplicates.

### Graph extraction

`ingestion/graph_builder.py` calls Claude Haiku once per chunk (concurrently, via an async
Neo4j driver) to extract entities and relationships, using tool-use to force a structured
response matching:

```
entities:      [{id, name, type}]
relationships: [{from, to, type, description}]
```

Entity types: `SERVICE`, `ERROR_CODE`, `RUNBOOK`, `INCIDENT`, `TEAM`, `DEPENDENCY`, `CONFIG`.
Relationship types: `CAUSED_BY`, `DEPENDS_ON`, `OWNED_BY`, `RESOLVED_BY`, `TRIGGERS`,
`MITIGATED_BY`.

- Entities are upserted as Neo4j nodes **labeled by their type** (e.g. `(:SERVICE {name: ...})`),
  `MERGE`d on `(label, name)` so the same real-world entity dedupes across chunks/documents
  even though the model invents a fresh local `id` on every call.
- Relationships are upserted as typed edges (e.g. `(:ERROR_CODE)-[:CAUSED_BY]->(:SERVICE)`)
  carrying `description` and `source_doc_id` properties.
- Every entity is linked to its source chunk via `(entity)-[:MENTIONED_IN]->(:Chunk {id})`.
  Neo4j only stores that `chunk_id` join key — `Neo4jGraphRetriever` hydrates the actual text
  by looking the id up in Chroma at query time, so graph traversal and vector/keyword search
  stay hybrid without duplicating content across stores.
- A chunk where extraction fails or returns no entities is skipped rather than failing the
  whole ingestion run.

## Retrieval

Each of the three retrievers implements the same `BaseRetriever.retrieve(query, top_k=10)`
interface and is fused by `ReciprocalRankFusionReranker`:

- **`ChromaVectorRetriever`** — embeds the query with the same `sentence-transformers`
  (`all-MiniLM-L6-v2`) model used at ingest time and does a cosine-similarity search.
- **`ElasticsearchKeywordRetriever`** — runs a `multi_match` BM25 query across `content`,
  `title`, `tags`, `service`, and separately calls Claude Haiku to extract structured filters
  from the query (`service`, `severity`, `error_code`, `date_range`), applying them as ES
  `filter` clauses (case-insensitive `term` filters for service/severity, a phrase filter on
  `content` for error codes since there's no dedicated field for them, and a `range` filter on
  `incident_date`). If extraction fails, it falls back to an unfiltered BM25 search rather than
  failing the query.
- **`Neo4jGraphRetriever`** — matches query terms against extracted entities and hydrates
  chunk content from Chroma (see Graph extraction above).

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
uv run pytest tests/integration      # spins up a real Neo4j via testcontainers (needs Docker)
RUN_INTEGRATION_TESTS=1 uv run pytest tests/eval  # requires docker compose services + ANTHROPIC_API_KEY
```
