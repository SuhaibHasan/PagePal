FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY api/ ./api/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY wiki/ ./wiki/

RUN uv pip install --system --no-cache .

FROM python:3.11-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY api/ ./api/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY wiki/ ./wiki/

USER app
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
