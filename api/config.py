from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "prod_docs"

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "prod_docs"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "prodsupportbuddy"

    redis_url: str = "redis://localhost:6379/0"

    api_cors_origins: str = "http://localhost:3000"

    # Gates POST /ingest and the wiki write endpoints (validate/delete) - see
    # api/auth.py. Empty (the default) leaves them open, e.g. for local dev or a
    # read-mostly public demo where /chat and the wiki reads still need to work
    # without a key.
    api_key: str = ""

    retrieval_top_k: int = 10
    rerank_top_k: int = 8

    # Only needed for POST /ingest requests targeting these sources.
    confluence_base_url: str = ""
    confluence_email: str = ""
    confluence_api_token: str = ""
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
