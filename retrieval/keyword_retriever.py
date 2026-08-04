from __future__ import annotations

from elasticsearch import Elasticsearch

from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult


class ElasticsearchKeywordRetriever(BaseRetriever):
    def __init__(self, client: Elasticsearch, index_name: str = "prod_docs") -> None:
        self._client = client
        self._index_name = index_name

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        response = self._client.search(
            index=self._index_name,
            query={"match": {"content": query}},
            size=top_k,
        )
        hits = response["hits"]["hits"]
        return [
            RetrievalResult(
                chunk_id=hit["_id"],
                document_id=hit["_source"].get("doc_id", hit["_id"]),
                content=hit["_source"].get("content", ""),
                score=hit["_score"],
                source_type="keyword",
                metadata={k: v for k, v in hit["_source"].items() if k != "content"},
            )
            for hit in hits
        ]

    def ping(self) -> bool:
        return bool(self._client.ping())
