from __future__ import annotations

import hashlib
from pathlib import Path

import anthropic

from ingestion.loaders.base import BaseLoader
from ingestion.loaders.pdf_parser import (
    ImageElement,
    TableElement,
    describe_image,
    extract_pdf_elements,
)
from ingestion.models import Document, SourceType

_EXTENSION_SOURCE_TYPES = {
    ".md": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
}


class FileLoader(BaseLoader):
    def __init__(
        self, root_dir: str | Path, anthropic_client: anthropic.Anthropic | None = None
    ) -> None:
        self.root_dir = Path(root_dir)
        # Only constructed on first actual use (describing a PDF's embedded
        # image) rather than eagerly here, so a purely-markdown or
        # image-free-PDF corpus never requires ANTHROPIC_API_KEY to be set.
        self._anthropic_client = anthropic_client

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.root_dir.rglob("*")):
            source_type = _EXTENSION_SOURCE_TYPES.get(path.suffix.lower())
            if not path.is_file() or source_type is None:
                continue

            relative_source = str(path.relative_to(self.root_dir))
            doc_id = hashlib.sha256(relative_source.encode()).hexdigest()[:16]
            title = path.stem
            url = path.resolve().as_uri()

            if source_type is SourceType.PDF:
                documents.extend(self._load_pdf(path, doc_id, title, url))
            else:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    documents.append(
                        Document(id=f"local-{doc_id}", title=title, content=content,
                                  source_type=source_type, url=url)
                    )
        return documents

    def _load_pdf(self, path: Path, doc_id: str, title: str, url: str) -> list[Document]:
        # A single parser can't do justice to a document that mixes prose,
        # tables, and images - pdfminer-style text extraction jumbles a
        # table's cells into unreadable runs of text, and silently drops
        # images entirely. extract_pdf_elements splits the page into the
        # three element kinds first, each handled by whichever parser is
        # actually suited to it; every element becomes its own Document so a
        # table's rows/columns (or an image's content) stay intact as their
        # own chunk instead of being diluted into - or lost from - the
        # surrounding prose.
        extraction = extract_pdf_elements(path)

        documents: list[Document] = []
        if extraction.text.strip():
            documents.append(
                Document(id=f"local-{doc_id}", title=title, content=extraction.text,
                          source_type=SourceType.PDF, url=url)
            )

        for index, table in enumerate(extraction.tables):
            documents.append(self._table_document(table, doc_id, title, url, index))

        for index, image in enumerate(extraction.images):
            document = self._image_document(image, doc_id, title, url, index)
            if document is not None:
                documents.append(document)

        return documents

    @staticmethod
    def _table_document(
        table: TableElement, doc_id: str, title: str, url: str, index: int
    ) -> Document:
        return Document(
            id=f"local-{doc_id}::table-{index}",
            title=f"{title} (table, page {table.page_number})",
            content=f"Table from page {table.page_number} of \"{title}\":\n\n{table.markdown}",
            source_type=SourceType.PDF,
            url=f"{url}#page={table.page_number}",
            extra={"element_type": "table", "page_number": table.page_number},
        )

    def _image_document(
        self, image: ImageElement, doc_id: str, title: str, url: str, index: int
    ) -> Document | None:
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic()
        description = describe_image(image, self._anthropic_client)
        if description is None:
            return None
        return Document(
            id=f"local-{doc_id}::image-{index}",
            title=f"{title} (image, page {image.page_number})",
            content=f"Image from page {image.page_number} of \"{title}\":\n\n{description}",
            source_type=SourceType.PDF,
            url=f"{url}#page={image.page_number}",
            extra={"element_type": "image", "page_number": image.page_number},
        )
