from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from ingestion.loaders.base import BaseLoader
from ingestion.models import Document, SourceType

_EXTENSION_SOURCE_TYPES = {
    ".md": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
}


class FileLoader(BaseLoader):
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.root_dir.rglob("*")):
            source_type = _EXTENSION_SOURCE_TYPES.get(path.suffix.lower())
            if not path.is_file() or source_type is None:
                continue
            content = self._read_text(path, source_type)
            if not content.strip():
                continue
            relative_source = str(path.relative_to(self.root_dir))
            doc_id = hashlib.sha256(relative_source.encode()).hexdigest()[:16]
            documents.append(
                Document(
                    id=f"local-{doc_id}",
                    title=path.stem,
                    content=content,
                    source_type=source_type,
                    url=path.resolve().as_uri(),
                )
            )
        return documents

    @staticmethod
    def _read_text(path: Path, source_type: SourceType) -> str:
        if source_type is SourceType.PDF:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.read_text(encoding="utf-8", errors="ignore")
