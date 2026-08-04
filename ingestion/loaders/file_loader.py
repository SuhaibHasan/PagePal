from __future__ import annotations

import hashlib
from pathlib import Path

from ingestion.loaders.base import BaseLoader
from ingestion.models import Document

DEFAULT_EXTENSIONS = (".md", ".txt", ".rst")


class FileLoader(BaseLoader):
    def __init__(
        self, root_dir: str | Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    ) -> None:
        self.root_dir = Path(root_dir)
        self.extensions = extensions

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            relative_source = str(path.relative_to(self.root_dir))
            doc_id = hashlib.sha256(relative_source.encode()).hexdigest()[:16]
            documents.append(
                Document(
                    id=doc_id,
                    source=relative_source,
                    title=path.stem,
                    content=content,
                    metadata={"path": str(path)},
                )
            )
        return documents
