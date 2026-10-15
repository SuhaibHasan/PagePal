from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx

from ingestion.loaders.file_loader import FileLoader
from ingestion.models import SourceType

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class _FakeMessages:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.messages = _FakeMessages(response=response, error=error)


def _image_description_client(text: str) -> _FakeAnthropicClient:
    return _FakeAnthropicClient(
        response=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    )


def test_file_loader_loads_markdown_files(tmp_path: Path):
    (tmp_path / "runbook.md").write_text("# Runbook\nRestart the payment-service.")
    (tmp_path / "ignored.txt").write_text("Investigate ERR-500 in payment-service.")
    (tmp_path / "ignored.json").write_text("{}")

    documents = FileLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].title == "runbook"
    assert documents[0].source_type == SourceType.MARKDOWN
    assert documents[0].url == (tmp_path / "runbook.md").resolve().as_uri()


def test_file_loader_assigns_stable_ids(tmp_path: Path):
    (tmp_path / "a.md").write_text("content")

    first_load = FileLoader(tmp_path).load()
    second_load = FileLoader(tmp_path).load()

    assert first_load[0].id == second_load[0].id


def test_file_loader_recurses_into_subdirectories(tmp_path: Path):
    nested = tmp_path / "runbooks" / "payments"
    nested.mkdir(parents=True)
    (nested / "oncall.md").write_text("Escalate to the payments team.")

    documents = FileLoader(tmp_path).load()

    assert documents[0].title == "oncall"


def test_file_loader_skips_files_with_no_extractable_text(tmp_path: Path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with (tmp_path / "blank.pdf").open("wb") as fh:
        writer.write(fh)

    # No anthropic_client passed: a blank page has no images either, so this
    # must not require ANTHROPIC_API_KEY to be set.
    documents = FileLoader(tmp_path).load()

    assert documents == []


def test_file_loader_splits_pdf_into_text_table_and_image_documents(tmp_path: Path):
    pdf_path = tmp_path / "runbook.pdf"
    pdf_path.write_bytes((FIXTURE_DIR / "table_and_image.pdf").read_bytes())
    client = _image_description_client("A diagram linking payment-service to postgres-primary.")

    documents = FileLoader(tmp_path, anthropic_client=client).load()

    assert len(documents) == 3
    by_element_type = {doc.extra.get("element_type"): doc for doc in documents}

    text_doc = next(doc for doc in documents if "element_type" not in doc.extra)
    assert "::" not in text_doc.id  # the plain-text document's id has no element suffix
    assert "Error code reference table" in text_doc.content
    assert "payments-infra" not in text_doc.content

    table_doc = by_element_type["table"]
    assert table_doc.id.endswith("::table-0")
    assert "| ERR-503 | P1 | payments-infra |" in table_doc.content
    assert table_doc.extra["page_number"] == 1

    image_doc = by_element_type["image"]
    assert image_doc.id.endswith("::image-0")
    assert "A diagram linking payment-service to postgres-primary." in image_doc.content
    assert image_doc.extra["page_number"] == 1


def test_file_loader_skips_image_document_when_description_fails(tmp_path: Path):
    pdf_path = tmp_path / "runbook.pdf"
    pdf_path.write_bytes((FIXTURE_DIR / "table_and_image.pdf").read_bytes())
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    client = _FakeAnthropicClient(error=error)

    documents = FileLoader(tmp_path, anthropic_client=client).load()

    element_types = {doc.extra.get("element_type") for doc in documents}
    assert "image" not in element_types
    assert "table" in element_types  # table extraction is unaffected by the image failure
