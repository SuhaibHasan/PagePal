from __future__ import annotations

from functools import lru_cache

import anthropic
import chromadb
import redis
from elasticsearch import Elasticsearch
from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase

from api.config import get_settings
from ingestion.embedders.embedder import SentenceTransformerEmbedder
from ingestion.graph_builder import GraphBuilder
from ingestion.pipeline import IngestionPipeline
from retrieval.answer_generator import AnswerGenerator
from retrieval.graph_retriever import Neo4jGraphRetriever
from retrieval.keyword_retriever import ElasticsearchKeywordRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_retriever import ChromaVectorRetriever


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


@lru_cache
def get_elasticsearch_client() -> Elasticsearch:
    settings = get_settings()
    return Elasticsearch(settings.elasticsearch_url)


@lru_cache
def get_neo4j_driver() -> Driver:
    settings = get_settings()
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


@lru_cache
def get_async_neo4j_driver() -> AsyncDriver:
    # Separate from get_neo4j_driver(): ingestion's graph builder needs an
    # async session, retrieval's graph traversal doesn't. Both point at the
    # same Neo4j instance.
    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache
def get_async_anthropic_client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


@lru_cache
def get_vector_retriever() -> ChromaVectorRetriever:
    settings = get_settings()
    return ChromaVectorRetriever(get_chroma_client(), get_embedder(), settings.chroma_collection)


@lru_cache
def get_keyword_retriever() -> ElasticsearchKeywordRetriever:
    settings = get_settings()
    return ElasticsearchKeywordRetriever(
        get_elasticsearch_client(),
        settings.elasticsearch_index,
        anthropic_client=get_anthropic_client(),
    )


@lru_cache
def get_graph_retriever() -> Neo4jGraphRetriever:
    settings = get_settings()
    collection = get_chroma_client().get_or_create_collection(settings.chroma_collection)
    return Neo4jGraphRetriever(
        get_neo4j_driver(),
        chroma_collection=collection,
        anthropic_client=get_anthropic_client(),
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    settings = get_settings()
    return AnswerGenerator(
        anthropic_client=get_anthropic_client(),
        async_anthropic_client=get_async_anthropic_client(),
        model=settings.anthropic_model,
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    graph_builder = GraphBuilder(
        get_async_neo4j_driver(), anthropic_client=get_async_anthropic_client()
    )
    return IngestionPipeline(
        chroma_client=get_chroma_client(),
        es_client=get_elasticsearch_client(),
        neo4j_driver=get_async_neo4j_driver(),
        chroma_collection=settings.chroma_collection,
        es_index=settings.elasticsearch_index,
        graph_builder=graph_builder,
    )
