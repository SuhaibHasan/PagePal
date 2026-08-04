from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "prod_support_chunks"

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "prod_support_chunks"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "prodsupportbuddy"

    redis_url: str = "redis://localhost:6379/0"

    api_cors_origins: str = "http://localhost:3000"

    retrieval_top_k: int = 8
    rerank_top_k: int = 5
    cache_ttl_seconds: int = 3600

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
