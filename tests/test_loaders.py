from pathlib import Path

from app.ingestion.loader import load_markdown, load_txt


def test_load_txt_returns_document_with_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("RAG 笔记", encoding="utf-8")

    document = load_txt(path)

    assert document.text == "RAG 笔记"
    assert document.metadata == {"source": str(path.resolve())}


def test_load_markdown_returns_document_with_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# 标题\n\n正文", encoding="utf-8")

    document = load_markdown(path)

    assert document.text == "# 标题\n\n正文"
    assert document.metadata == {"source": str(path.resolve())}
