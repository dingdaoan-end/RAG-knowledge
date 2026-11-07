from io import BytesIO
from pathlib import Path
import re

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.ingestion import loader


def make_pdf(*page_texts: str | None, password: str | None = None) -> bytes:
    """Create real, self-contained PDFs without external fixtures or parsers."""

    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md", "notes.markdown", "NOTES.MD"])
def test_text_upload_preserves_utf8_text_and_only_source_metadata(filename: str) -> None:
    documents = loader.parse_upload("# 标题\n\n正文\n".encode("utf-8"), filename)

    assert len(documents) == 1
    assert documents[0].text == "# 标题\n\n正文\n"
    assert documents[0].metadata == {"source": filename}


def test_text_upload_accepts_utf8_bom() -> None:
    documents = loader.parse_upload("中文正文".encode("utf-8-sig"), "notes.txt")

    assert documents[0].text == "中文正文"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (r"C:\fakepath\report.txt", "report.txt"),
        ("../../private/notes.md", "notes.md"),
        (r"..\secret/mixed\report.markdown", "report.markdown"),
        ("/private/上传笔记.txt", "上传笔记.txt"),
    ],
)
def test_upload_uses_basename_without_client_paths(filename: str, expected: str) -> None:
    assert loader.display_filename(filename) == expected
    documents = loader.parse_upload(b"readable text", filename)

    assert documents[0].metadata == {"source": expected}


@pytest.mark.parametrize(
    "filename",
    ["", " ", ".", "..", "/", "path/", "bad\nname.txt", "bad\x00name.txt", "bad\x7fname.txt", "bad\x85name.txt", "bad\rdir/report.txt", "a" * 256 + ".txt"],
)
def test_invalid_filenames_are_rejected_before_path_cleanup(filename: str) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.display_filename(filename)

    assert caught.value.status_code == 422
    assert isinstance(caught.value, ValueError)
    assert "文件名" in str(caught.value)


@pytest.mark.parametrize("filename", ["notes.docx", "image.png", "no-extension", "archive.pdf.exe"])
def test_unsupported_extension_returns_415(filename: str) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(b"not used", filename)

    assert caught.value.status_code == 415
    assert filename not in str(caught.value)


@pytest.mark.parametrize("content", [b"", b" \r\n\t", b"\xef\xbb\xbf", "\u3000\n".encode("utf-8")])
def test_empty_or_whitespace_text_is_rejected(content: bytes) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(content, "empty.txt")

    assert caught.value.status_code == 422


@pytest.mark.parametrize("content", [b"\xff\xfeabc", "GBK文本".encode("gbk")])
def test_text_requires_valid_utf8(content: bytes) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(content, "private.txt")

    assert caught.value.status_code == 422
    assert "UTF-8" in str(caught.value)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("content", [b"hello\x00world", b"\x01\x02\x03binary", b"PK\x03\x04zip"])
def test_binary_content_is_rejected_with_a_text_extension(content: bytes) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(content, "pretend.md")

    assert caught.value.status_code == 422
    assert "二进制" in str(caught.value)


@pytest.mark.parametrize("filename", ["pretend.txt", "pretend.md", "pretend.markdown"])
def test_pdf_bytes_cannot_be_uploaded_with_a_text_extension(filename: str) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(make_pdf("Hidden PDF"), filename)

    assert caught.value.status_code == 422
    assert "扩展名" in str(caught.value)


def test_pdf_upload_extracts_real_text_and_preserves_original_page_numbers() -> None:
    documents = loader.parse_upload(make_pdf("First page", None, "Third page"), r"C:\fakepath\report.PDF")

    assert [document.text.strip() for document in documents] == ["First page", "Third page"]
    assert [document.metadata for document in documents] == [
        {"source": "report.PDF", "page": 1},
        {"source": "report.PDF", "page": 3},
    ]


@pytest.mark.parametrize("content", [b"", b"plain text", b"\n%PDF-1.7\n", b"%PDX-1.7\n"])
def test_pdf_extension_requires_matching_pdf_signature(content: bytes) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(content, "report.pdf")

    assert caught.value.status_code == 422


def test_damaged_pdf_returns_a_safe_error_without_parser_details() -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(b"%PDF-1.7\nSENSITIVE-MARKER\n%%EOF", "private-report.pdf")

    assert caught.value.status_code == 422
    assert "PDF" in str(caught.value)
    assert "SENSITIVE-MARKER" not in str(caught.value)
    assert "private-report" not in str(caught.value)
    assert "xref" not in str(caught.value)


def test_pdf_parser_rejects_broken_xref_instead_of_repairing_it() -> None:
    content = re.sub(rb"startxref\n\d+", b"startxref\n1", make_pdf("Actual text"))

    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(content, "broken.pdf")

    assert caught.value.status_code == 422


@pytest.mark.parametrize("page_texts", [(None,), (None, None), ("   ",)])
def test_pdf_without_extractable_text_explains_that_ocr_is_needed(page_texts: tuple[str | None, ...]) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(make_pdf(*page_texts), "scan.pdf")

    assert caught.value.status_code == 422
    assert "没有可提取正文" in str(caught.value)
    assert "OCR" in str(caught.value)
    assert "当前未支持" in str(caught.value)


@pytest.mark.parametrize("password", ["secret", ""])
def test_encrypted_pdf_is_rejected_even_with_an_empty_password(password: str) -> None:
    with pytest.raises(loader.DocumentLoadError) as caught:
        loader.parse_upload(make_pdf("Private text", password=password), "encrypted.pdf")

    assert caught.value.status_code == 422
    assert "加密" in str(caught.value)
    assert "Private text" not in str(caught.value)


def test_load_pdf_reads_only_the_provided_file_and_keeps_absolute_source(tmp_path: Path) -> None:
    path = tmp_path / "local.pdf"
    path.write_bytes(make_pdf("Local text", None, "Final page"))

    documents = loader.load_pdf(path)

    assert [document.text.strip() for document in documents] == ["Local text", "Final page"]
    assert [document.metadata for document in documents] == [
        {"source": str(path.resolve()), "page": 1},
        {"source": str(path.resolve()), "page": 3},
    ]
