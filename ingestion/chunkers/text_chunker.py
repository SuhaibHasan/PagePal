from __future__ import annotations

import tiktoken

from ingestion.chunkers.base import BaseChunker
from ingestion.models import Chunk, Document


class TextChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = self._encoding.encode(document.content)
        if not tokens:
            return []

        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(tokens), stride)):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            text = self._encoding.decode(window)
            chunks.append(
                Chunk(
                    id=f"{document.id}::{index}",
                    document_id=document.id,
                    content=text,
                    metadata={**document.metadata, "source": document.source, "chunk_index": index},
                )
            )
            if start + self.chunk_size >= len(tokens):
                break
        return chunks
