from pathlib import Path

from ingestion.loaders.file_loader import FileLoader


def test_file_loader_loads_matching_files(tmp_path: Path):
    (tmp_path / "runbook.md").write_text("# Runbook\nRestart the payment-service.")
    (tmp_path / "notes.txt").write_text("Investigate ERR-500 in payment-service.")
    (tmp_path / "ignored.json").write_text("{}")

    documents = FileLoader(tmp_path).load()

    sources = {document.source for document in documents}
    assert sources == {"runbook.md", "notes.txt"}


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

    assert documents[0].source == str(Path("runbooks") / "payments" / "oncall.md")
