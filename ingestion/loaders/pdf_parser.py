from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import pdfplumber
import pymupdf

logger = logging.getLogger(__name__)

# Matches the model id used elsewhere in this codebase (ingestion/graph_builder.py,
# wiki/distiller.py, wiki/retriever.py, wiki/writer.py) for Haiku extraction tasks -
# Haiku is multimodal, so the same model does image description too.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_DESCRIPTION_TOKENS = 300

IMAGE_DESCRIPTION_SYSTEM_PROMPT = (
    "You are looking at an image embedded in a production support runbook or "
    "postmortem. It may be an architecture diagram, a dashboard/metrics "
    "screenshot, a terminal/log screenshot, or a flowchart. Write a dense, "
    "literal, searchable description: name every service/component and label "
    "visible, describe the relationships or data shown, and transcribe any "
    "legible text (error messages, numbers, codes). Do not speculate about "
    "anything not visibly present. Plain text only, no markdown."
)

# PDF -> raster image formats Claude's vision input accepts directly.
_SUPPORTED_IMAGE_EXTENSIONS = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


@dataclass
class TableElement:
    page_number: int
    markdown: str


@dataclass
class ImageElement:
    page_number: int
    image_bytes: bytes
    media_type: str


@dataclass
class PdfExtraction:
    text: str
    tables: list[TableElement] = field(default_factory=list)
    images: list[ImageElement] = field(default_factory=list)


def extract_pdf_elements(path: Path) -> PdfExtraction:
    """Splits a PDF into prose text, tables (as markdown), and raw images.

    Uses pdfplumber for layout-aware table detection (and to extract the
    remaining prose text with table regions excluded, so a table's cells
    don't also show up jumbled into the surrounding text), and pymupdf to
    pull embedded images as ready-to-use bytes - each library for the one
    thing it's actually good at, rather than forcing one parser to do a job
    it's not suited for.
    """
    tables: list[TableElement] = []
    text_parts: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_tables = page.find_tables()
            for table in page_tables:
                markdown = _table_to_markdown(table.extract())
                if markdown:
                    tables.append(TableElement(page_number=page_number, markdown=markdown))

            table_bboxes = [table.bbox for table in page_tables]
            text_page = page.filter(
                lambda obj, bboxes=table_bboxes: _not_in_any_bbox(obj, bboxes)
            )
            page_text = text_page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)

    images = _extract_images(path)

    return PdfExtraction(text="\n\n".join(text_parts), tables=tables, images=images)


def _not_in_any_bbox(obj: dict, bboxes: list[tuple[float, float, float, float]]) -> bool:
    for x0, top, x1, bottom in bboxes:
        if obj["x0"] >= x0 - 1 and obj["x1"] <= x1 + 1 and obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1:
            return False
    return True


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    cleaned = [[(cell or "").replace("\n", " ").replace("|", "\\|").strip() for cell in row] for row in rows]
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_images(path: Path) -> list[ImageElement]:
    images: list[ImageElement] = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                try:
                    extracted = doc.extract_image(xref)
                except Exception:
                    logger.warning(
                        "Failed to extract image xref=%s on page %d of %s",
                        xref, page_number, path, exc_info=True,
                    )
                    continue

                media_type = _SUPPORTED_IMAGE_EXTENSIONS.get(extracted["ext"])
                if media_type is None:
                    # e.g. an embedded CMYK/raw image pymupdf couldn't normalize to
                    # a web-safe format - skip it rather than send Claude bytes it
                    # can't decode.
                    continue

                images.append(
                    ImageElement(
                        page_number=page_number,
                        image_bytes=extracted["image"],
                        media_type=media_type,
                    )
                )
    return images


def describe_image(
    image: ImageElement, anthropic_client: anthropic.Anthropic
) -> str | None:
    """Returns a searchable text description of one image, or None if Claude
    couldn't produce one - the caller skips that image rather than failing
    the whole document."""
    try:
        response = anthropic_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_DESCRIPTION_TOKENS,
            system=IMAGE_DESCRIPTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.media_type,
                                "data": base64.b64encode(image.image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
        )
    except anthropic.APIError:
        logger.warning("Image description failed for page %d", image.page_number, exc_info=True)
        return None

    text_block = next((block for block in response.content if block.type == "text"), None)
    if text_block is None or not text_block.text.strip():
        return None
    return text_block.text.strip()
