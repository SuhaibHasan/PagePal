# ProdSupportBuddy

A RAG-powered assistant that helps engineers resolve production incidents by retrieving
relevant runbooks, postmortems, and prior incidents from a hybrid vector/keyword/graph
knowledge base and answering with a Claude model.

## Structure

```
ingestion/    loaders, chunkers, embedders, and an LLM-based Neo4j graph builder
retrieval/    vector (Chroma), keyword (Elasticsearch), and graph (Neo4j) retrievers,
              a cross-encoder reranker, and the streaming answer generator
api/          FastAPI service: streaming /chat, /ingest, /graph/explore, /health
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

### Endpoints

**`POST /chat`** — `{ session_id?, message, filters?: { service?, severity? } }`, streamed back as
Server-Sent Events:

```
event: session
data: {"session_id": "..."}

event: token
data: {"text": "Restart "}

event: token
data: {"text": "the payment-service pods."}

event: citations
data: {"citations": [{"title": "...", "url": "...", "retrieval_path": "graph+keyword", "graph_path": "..."}]}

event: done
data: {}
```

A missing `session_id` is generated server-side and echoed back in the first event so the
client can persist it for follow-up turns. `filters.service`/`filters.severity` are passed to
the keyword retriever and take precedence over whatever it infers from the query text itself
(see Retrieval below). Conversation history is stored in Redis per `session_id` with a 1-hour
TTL, and the last 5 turns are included in every prompt for follow-up context. Each session is
rate-limited to 10 requests/minute via a Redis sorted-set sliding window; requests beyond that
get `429`.

**`POST /ingest`** — `{ source_type: "local"|"confluence"|"pagerduty"|"jira", path?, space_key?, jql? }`
kicks off ingestion + graph building for one source as a FastAPI background task and returns
`202` immediately (`path` is required for `local`/`pagerduty`; Confluence/Jira credentials come
from `.env`).

**`GET /graph/explore?entity_name=auth-service`** — returns the 2-hop subgraph around a named
entity as `{ nodes: [{id, name, labels}], edges: [{source, target, type}] }` for frontend graph
visualization (no LLM call — the entity name is taken literally, unlike `/chat`'s retrieval).

**`GET /health`** — pings ChromaDB, Elasticsearch, Neo4j, and Redis; `200` with per-service
`ok`/`error: ...` status and an overall `ok`/`degraded`.

### Tracing

Every `/chat` request is wrapped in nested OpenTelemetry spans: `query_routing` (the whole
request) → `retrieval.vector` / `retrieval.keyword` / `retrieval.graph` (run concurrently) →
`reranking` → `llm_call`. Spans print to the console by default (`api/telemetry.py`) — swap in
an OTLP exporter there to ship to a real collector.

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
The same loaders and credentials back `POST /ingest`.

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

Each of the three retrievers implements the same
`BaseRetriever.retrieve(query, top_k=10, filters=None)` interface and is fused by
`CrossEncoderReranker`. `filters` is only meaningful to the keyword retriever (vector/graph
accept and ignore it); it's how `POST /chat`'s `filters.service`/`filters.severity` reach ES.

- **`ChromaVectorRetriever`** — embeds the query with the same `sentence-transformers`
  (`all-MiniLM-L6-v2`) model used at ingest time and does a cosine-similarity search.
- **`ElasticsearchKeywordRetriever`** — runs a `multi_match` BM25 query across `content`,
  `title`, `tags`, `service`, and separately calls Claude Haiku to extract structured filters
  from the query (`service`, `severity`, `error_code`, `date_range`), applying them as ES
  `filter` clauses (case-insensitive `term` filters for service/severity, a phrase filter on
  `content` for error codes since there's no dedicated field for them, and a `range` filter on
  `incident_date`). Explicit `filters` from the request override the LLM's guess for
  service/severity. If extraction fails, it falls back to an unfiltered BM25 search rather than
  failing the query.
- **`Neo4jGraphRetriever`** — calls Claude Haiku to extract the services, error codes, and
  incident identifiers mentioned in the query, then traverses up to 2 hops from those entities
  in the semantic graph (`MATCH (n)-[r*1..2]-(m) WHERE toLower(n.name) IN $entities ...`,
  excluding `MENTIONED_IN` hops so it stays in the entity graph rather than hopping through
  chunks that happen to share an unrelated entity). Each subgraph path is serialized into a
  readable arrow chain, e.g. `auth-service → DEPENDS_ON → redis-cache → CAUSED_BY →
  INCIDENT-4521`, and every node along a path is resolved back to its source chunk_id and
  hydrated from Chroma. Each returned chunk's content is prefixed with the path(s) that led to
  it, so the LLM sees both the causal reasoning chain and the supporting text, tagged
  `source_type="graph"`. It also exposes `explore_subgraph(entity_name)`, used by
  `GET /graph/explore` — the same 2-hop/no-`MENTIONED_IN` traversal, but keyed on a literal
  entity name (no LLM extraction) and returned as raw nodes/edges rather than serialized paths
  or hydrated chunks, since that's what a visualization needs instead.

### Reranking

`CrossEncoderReranker` takes each retriever's results (up to 10 each), deduplicates by
`chunk_id` — merging metadata and keeping the richer content when the same chunk was found by
more than one retriever (e.g. the graph retriever's path-prefixed content wins over a plain
vector hit) — and combines their `source_type` tags (`"vector"`, `"keyword+vector"`,
`"graph+keyword+vector"`, ...). The deduplicated candidates are then scored directly against
the query with `cross-encoder/ms-marco-MiniLM-L-6-v2` (`sentence-transformers`, fully local,
no external API) and the top 8 by that score become the unified context passed to the LLM,
each still carrying its provenance tag and metadata.

### Answer generation

`AnswerGenerator` (`retrieval/answer_generator.py`) builds the final prompt to Claude Sonnet
from the reranked context: each of the top 8 chunks becomes `[Source N]: <content>`, and any
`graph_paths` carried in their metadata are collected (deduped) into a separate `[Graph
Context]:` section so the model can reason over causal chains distinctly from raw evidence
text. Its system prompt instructs the model to answer only from context and cite sources as
`[Source N]`; after the answer streams back, those citation markers are parsed out of the text
and mapped back to each chunk's `title`, `url`, and `retrieval_path` (the same provenance tag
the reranker produced, e.g. `"graph+keyword"`), plus `graph_path` when the cited chunk came
from the graph retriever. Only sources the model actually cited are returned — not every chunk
that was in context.

`astream_answer(query, context, history)` (used by `POST /chat`) streams token-by-token via
`AsyncAnthropic.messages.stream(...)` for real-time SSE delivery, prepending up to 5 prior
`(user, assistant)` turns from Redis before the new message so follow-up questions keep
context. `generate(...)` is the synchronous, non-streaming equivalent (used by tests and
anything that just wants the final answer back).

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
