import pytest

from app.ingestion.document import Document
from app.ingestion.splitter import (
    character_split,
    paragraph_split,
    recursive_split,
)


SPLITTERS = [character_split, paragraph_split, recursive_split]


@pytest.mark.parametrize("splitter", SPLITTERS)
def test_empty_text_returns_no_chunks(splitter) -> None:
    document = Document(text="", metadata={"source": "empty.txt"})

    assert splitter(document, chunk_size=10, chunk_overlap=2) == []


@pytest.mark.parametrize("splitter", SPLITTERS)
def test_short_text_returns_one_chunk_with_metadata(splitter) -> None:
    document = Document(text="短文本", metadata={"source": "notes.txt", "topic": "RAG"})

    chunks = splitter(document, chunk_size=10, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == ["短文本"]
    assert chunks[0].metadata == {
        "source": "notes.txt",
        "topic": "RAG",
        "chunk_id": 0,
    }


def test_character_split_handles_long_text_and_overlap() -> None:
    document = Document(text="甲乙丙丁戊己庚辛壬癸子", metadata={"source": "demo.txt"})

    chunks = character_split(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "甲乙丙丁",
        "丁戊己庚",
        "庚辛壬癸",
        "癸子",
    ]
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [0, 1, 2, 3]


def test_paragraph_split_combines_without_breaking_paragraphs() -> None:
    document = Document(
        text="第一段\n\n第二段\n\n第三段",
        metadata={"source": "article.md"},
    )

    chunks = paragraph_split(document, chunk_size=10, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == ["第一段\n\n第二段\n\n", "第三段"]


def test_paragraph_split_falls_back_for_one_long_paragraph() -> None:
    document = Document(text="甲乙丙丁戊己庚辛壬癸", metadata={})

    chunks = paragraph_split(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == ["甲乙丙丁", "丁戊己庚", "庚辛壬癸"]


def test_recursive_split_uses_chinese_sentence_boundaries() -> None:
    document = Document(text="甲乙丙。丁戊己！庚辛壬？", metadata={"source": "story.txt"})

    chunks = recursive_split(document, chunk_size=7, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == ["甲乙丙。", "丁戊己！", "庚辛壬？"]


@pytest.mark.parametrize("splitter", [character_split, recursive_split])
def test_internal_whitespace_is_not_deleted(splitter) -> None:
    document = Document(text="AB    CD", metadata={})

    chunks = splitter(document, chunk_size=2, chunk_overlap=0)

    assert "".join(chunk.text for chunk in chunks) == document.text


def test_paragraph_overlap_survives_long_paragraph_fallback() -> None:
    document = Document(text="AAAA\n\nBBBBBBBB\n\nCCC", metadata={})

    chunks = paragraph_split(document, chunk_size=6, chunk_overlap=1)

    assert all(len(chunk.text) <= 6 for chunk in chunks)
    assert all(
        previous.text[-1:] == current.text[:1]
        for previous, current in zip(chunks, chunks[1:])
    )


def test_chunk_metadata_is_copied_not_shared() -> None:
    document = Document(
        text="甲乙丙丁戊己",
        metadata={"source": "demo.txt", "tags": ["RAG"]},
    )

    chunks = character_split(document, chunk_size=4, chunk_overlap=1)
    chunks[0].metadata["changed"] = True
    chunks[0].metadata["tags"].append("changed")

    assert "changed" not in document.metadata
    assert "changed" not in chunks[1].metadata
    assert document.metadata["tags"] == ["RAG"]
    assert chunks[1].metadata["tags"] == ["RAG"]


@pytest.mark.parametrize("splitter", SPLITTERS)
@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_parameters_raise_value_error(
    splitter, chunk_size: int, chunk_overlap: int
) -> None:
    document = Document(text="文本", metadata={})

    with pytest.raises(ValueError):
        splitter(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@pytest.mark.parametrize("splitter", SPLITTERS)
@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(4.5, 1), (True, 0), (4, 1.5), (4, False)],
)
def test_parameters_must_be_integers(splitter, chunk_size, chunk_overlap) -> None:
    with pytest.raises(TypeError):
        splitter(
            Document(text="文本", metadata={}),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
