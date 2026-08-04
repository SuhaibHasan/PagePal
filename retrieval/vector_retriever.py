from __future__ import annotations

import chromadb

from ingestion.embedders.base import BaseEmbedder
from retrieval.base import BaseRetriever
from retrieval.models import RetrievalResult


class ChromaVectorRetriever(BaseRetriever):
    def __init__(
        self,
        client: chromadb.ClientAPI,
        embedder: BaseEmbedder,
        collection_name: str = "prod_support_chunks",
    ) -> None:
        self._embedder = embedder
        self._collection = client.get_or_create_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed([query])[0]
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                document_id=(metadata or {}).get("document_id", chunk_id.split("::")[0]),
                content=content,
                score=1.0 - distance,
                source_type="vector",
                metadata=metadata or {},
            )
            for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def ping(self) -> None:
        self._collection.count()
