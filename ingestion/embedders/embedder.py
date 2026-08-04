from __future__ import annotations

from sentence_transformers import SentenceTransformer

from ingestion.embedders.base import BaseEmbedder

MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(BaseEmbedder):
    # Fully local, no external API - the same model must be used at query time
    # so document and query vectors land in the same embedding space.
    dimension = 384

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
