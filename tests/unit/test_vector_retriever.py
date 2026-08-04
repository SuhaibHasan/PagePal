import pytest

from ingestion.embedders.base import BaseEmbedder
from retrieval.vector_retriever import ChromaVectorRetriever


class _FakeEmbedder(BaseEmbedder):
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeCollection:
    def __init__(self, query_result: dict) -> None:
        self._query_result = query_result
        self.query_calls: list[dict] = []

    def query(self, query_embeddings, n_results):
        self.query_calls.append({"query_embeddings": query_embeddings, "n_results": n_results})
        return self._query_result

    def count(self) -> int:
        return 0


class _FakeChromaClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection
        self.get_or_create_collection_calls: list[dict] = []

    def get_or_create_collection(self, name, metadata=None):
        self.get_or_create_collection_calls.append({"name": name, "metadata": metadata})
        return self._collection


def make_query_result() -> dict:
    return {
        "ids": [["doc-1::0", "doc-1::1"]],
        "documents": [["First chunk text", "Second chunk text"]],
        "metadatas": [
            [{"doc_id": "doc-1", "title": "Runbook"}, {"doc_id": "doc-1", "title": "Runbook"}]
        ],
        "distances": [[0.1, 0.4]],
    }


def test_retrieve_embeds_query_and_passes_it_to_the_collection():
    collection = _FakeCollection(make_query_result())
    client = _FakeChromaClient(collection)
    embedder = _FakeEmbedder()
    retriever = ChromaVectorRetriever(client, embedder, collection_name="prod_docs")

    retriever.retrieve("why is payment-service failing", top_k=10)

    assert embedder.calls == [["why is payment-service failing"]]
    assert collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2, 0.3]]


def test_retrieve_defaults_to_ten_results():
    collection = _FakeCollection(make_query_result())
    client = _FakeChromaClient(collection)
    retriever = ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    retriever.retrieve("query")

    assert collection.query_calls[0]["n_results"] == 10


def test_collection_created_with_cosine_space():
    collection = _FakeCollection(make_query_result())
    client = _FakeChromaClient(collection)

    ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    assert client.get_or_create_collection_calls[0] == {
        "name": "prod_docs",
        "metadata": {"hnsw:space": "cosine"},
    }


def test_retrieve_maps_distance_to_similarity_score_and_metadata():
    collection = _FakeCollection(make_query_result())
    client = _FakeChromaClient(collection)
    retriever = ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    results = retriever.retrieve("query")

    assert len(results) == 2
    first, second = results
    assert first.chunk_id == "doc-1::0"
    assert first.document_id == "doc-1"
    assert first.content == "First chunk text"
    assert first.score == pytest.approx(0.9)
    assert first.source_type == "vector"
    assert first.metadata == {"doc_id": "doc-1", "title": "Runbook"}
    assert second.score == pytest.approx(0.6)


def test_retrieve_falls_back_to_chunk_id_prefix_when_metadata_missing():
    query_result = {
        "ids": [["doc-9::2"]],
        "documents": [["text"]],
        "metadatas": [[None]],
        "distances": [[0.2]],
    }
    collection = _FakeCollection(query_result)
    client = _FakeChromaClient(collection)
    retriever = ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    results = retriever.retrieve("query")

    assert results[0].document_id == "doc-9"
    assert results[0].metadata == {}


def test_retrieve_returns_empty_list_when_collection_has_no_matches():
    empty_result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    collection = _FakeCollection(empty_result)
    client = _FakeChromaClient(collection)
    retriever = ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    assert retriever.retrieve("query") == []


def test_ping_calls_collection_count():
    calls = []

    class _CountingCollection(_FakeCollection):
        def count(self) -> int:
            calls.append(1)
            return 42

    client = _FakeChromaClient(_CountingCollection(make_query_result()))
    retriever = ChromaVectorRetriever(client, _FakeEmbedder(), collection_name="prod_docs")

    retriever.ping()

    assert calls == [1]
