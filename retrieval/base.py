from abc import ABC, abstractmethod

from retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, str] | None = None
    ) -> list[RetrievalResult]:
        raise NotImplementedError
