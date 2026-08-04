from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ingestion.embedders.base import BaseEmbedder


class DefaultEmbedder(BaseEmbedder):
    # Local ONNX MiniLM model bundled with chromadb - no external API key required.
    dimension = 384

    def __init__(self) -> None:
        self._embedding_function = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._embedding_function(texts)]
