from pathlib import Path

from ingestion.loaders.file_loader import FileLoader
from ingestion.models import SourceType


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

    documents = FileLoader(tmp_path).load()

    assert documents == []
