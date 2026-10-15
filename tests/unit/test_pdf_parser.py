from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx

from ingestion.loaders.pdf_parser import (
    ImageElement,
    _table_to_markdown,
    describe_image,
    extract_pdf_elements,
)

FIXTURE = Path(__file__).parent / "fixtures" / "table_and_image.pdf"


def _text_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.messages = _FakeMessages(response=response, error=error)


def test_extract_pdf_elements_separates_text_table_and_images():
    extraction = extract_pdf_elements(FIXTURE)

    assert "payments-infra" not in extraction.text  # table content excluded from prose
    assert "ERR-429" not in extraction.text  # only appears inside the table
    assert "Error code reference table" in extraction.text  # surrounding prose kept
    assert "More prose after the image and table" in extraction.text

    assert len(extraction.tables) == 1
    table = extraction.tables[0]
    assert table.page_number == 1
    assert "| Error Code | Severity | Owning Team |" in table.markdown
    assert "| ERR-503 | P1 | payments-infra |" in table.markdown

    assert len(extraction.images) == 1
    image = extraction.images[0]
    assert image.page_number == 1
    assert image.media_type == "image/png"
    assert len(image.image_bytes) > 0


def test_table_to_markdown_handles_none_cells_and_pipes():
    rows = [["A", "B"], [None, "has | pipe"]]
    markdown = _table_to_markdown(rows)
    assert markdown == "| A | B |\n| --- | --- |\n|  | has \\| pipe |"


def test_table_to_markdown_empty_rows_returns_empty_string():
    assert _table_to_markdown([]) == ""


def test_describe_image_returns_text_on_success():
    client = _FakeClient(response=_text_response("A diagram of two services."))
    image = ImageElement(page_number=1, image_bytes=b"fake-png-bytes", media_type="image/png")

    description = describe_image(image, client)

    assert description == "A diagram of two services."
    request = client.messages.calls[0]
    content = request["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"


def test_describe_image_returns_none_on_api_error():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    client = _FakeClient(error=error)
    image = ImageElement(page_number=1, image_bytes=b"fake-png-bytes", media_type="image/png")

    assert describe_image(image, client) is None


def test_describe_image_returns_none_on_empty_response():
    client = _FakeClient(response=_text_response("   "))
    image = ImageElement(page_number=1, image_bytes=b"fake-png-bytes", media_type="image/png")

    assert describe_image(image, client) is None
