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
        collection_name: str = "prod_docs",
    ) -> None:
        # embedder is injected (SentenceTransformerEmbedder / all-MiniLM-L6-v2 in
        # production) rather than hardcoded here, since query-time embeddings must
        # come from the same model used to embed chunks at ingest time.
        self._embedder = embedder
        # Embeddings are unit-normalized, so cosine is the intended metric; this
        # only takes effect if this call is what creates the collection, but it
        # makes the distance-derived score below meaningful either way.
        self._collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

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
                document_id=(metadata or {}).get("doc_id", chunk_id.split("::")[0]),
                content=content,
                # Cosine distance -> similarity, so higher is better here too,
                # consistent with the keyword (BM25) and graph (hit count) scores.
                score=1.0 - distance,
                source_type="vector",
                metadata=metadata or {},
            )
            for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def ping(self) -> None:
        self._collection.count()
