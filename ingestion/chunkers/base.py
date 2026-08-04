from abc import ABC, abstractmethod

from ingestion.models import Chunk, Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
