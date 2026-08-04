from __future__ import annotations

from functools import lru_cache

import chromadb
import redis
from elasticsearch import Elasticsearch
from neo4j import Driver, GraphDatabase

from api.config import get_settings
from ingestion.embedders.embedder import DefaultEmbedder
from retrieval.graph_retriever import Neo4jGraphRetriever
from retrieval.keyword_retriever import ElasticsearchKeywordRetriever
from retrieval.reranker import ReciprocalRankFusionReranker
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
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_embedder() -> DefaultEmbedder:
    return DefaultEmbedder()


@lru_cache
def get_vector_retriever() -> ChromaVectorRetriever:
    settings = get_settings()
    return ChromaVectorRetriever(get_chroma_client(), get_embedder(), settings.chroma_collection)


@lru_cache
def get_keyword_retriever() -> ElasticsearchKeywordRetriever:
    settings = get_settings()
    return ElasticsearchKeywordRetriever(get_elasticsearch_client(), settings.elasticsearch_index)


@lru_cache
def get_graph_retriever() -> Neo4jGraphRetriever:
    return Neo4jGraphRetriever(get_neo4j_driver())


@lru_cache
def get_reranker() -> ReciprocalRankFusionReranker:
    return ReciprocalRankFusionReranker()
