from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunkers.base import BaseChunker
from ingestion.models import Chunk, Document

DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 50


class TextChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._splitter.split_text(document.content)
        return [
            Chunk(
                id=f"{document.id}::{index}",
                doc_id=document.id,
                chunk_index=index,
                content=text,
                title=document.title,
                source_type=document.source_type,
                url=document.url,
                incident_date=document.incident_date,
                severity=document.severity,
                service_tags=document.service_tags,
                extra=document.extra,
            )
            for index, text in enumerate(texts)
        ]
