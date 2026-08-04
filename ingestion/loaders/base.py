from abc import ABC, abstractmethod

from ingestion.models import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]:
        raise NotImplementedError
