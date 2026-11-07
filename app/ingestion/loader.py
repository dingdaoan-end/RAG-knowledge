from io import BytesIO
import ntpath
from pathlib import Path
import unicodedata

from pypdf import PdfReader

from .document import Document


ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


class DocumentLoadError(ValueError):
    """可安全展示给上传用户的解析错误。"""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def display_filename(filename: str) -> str:
    """仅保留用于展示的文件名，不使用客户端路径访问文件。"""

    if not isinstance(filename, str) or any(
        unicodedata.category(character) == "Cc" for character in filename
    ):
        raise DocumentLoadError("文件名无效。")
    name = ntpath.basename(filename.replace("\\", "/")).strip()
    if not name or name in {".", ".."} or len(name) > 255:
        raise DocumentLoadError("文件名无效。")
    return name


def _parse_pdf(content: bytes, source: str) -> list[Document]:
    if not content.startswith(b"%PDF-"):
        raise DocumentLoadError("文件内容与 PDF 扩展名不符。")

    documents = []
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise DocumentLoadError("不支持加密 PDF，请先解除加密后上传。")
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(text=text, metadata={"source": source, "page": page_number})
                )
    except DocumentLoadError:
        raise
    except Exception:
        # Parser exceptions can contain file content or internal implementation details.
        raise DocumentLoadError("PDF 文件损坏或格式无效。") from None

    if not documents:
        raise DocumentLoadError("PDF 没有可提取正文，扫描 PDF 需要 OCR，当前未支持。")
    return documents


def parse_upload(content: bytes, filename: str) -> list[Document]:
    """从上传字节解析正文，仅将安全文件名写入来源元数据。"""

    source = display_filename(filename)
    extension = Path(source).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise DocumentLoadError("不支持此文件类型，仅支持 TXT、Markdown 和 PDF。", status_code=415)
    if not content:
        raise DocumentLoadError("文件内容为空。")
    if extension == ".pdf":
        return _parse_pdf(content, source)
    if content.startswith(b"%PDF-"):
        raise DocumentLoadError("文件内容与扩展名不符。")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DocumentLoadError("文本文件必须使用 UTF-8 编码。") from None
    if text.lstrip().startswith("%PDF-"):
        raise DocumentLoadError("文件内容与扩展名不符。")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r\f"
        for character in text
    ):
        raise DocumentLoadError("文本文件包含二进制内容。")
    if not text.strip():
        raise DocumentLoadError("文本文件没有有效正文。")
    return [Document(text=text, metadata={"source": source})]


def load_pdf(path: str | Path) -> list[Document]:
    """读取本地 PDF，并以实际页码和绝对路径记录来源。"""

    file_path = Path(path)
    return _parse_pdf(file_path.read_bytes(), str(file_path.resolve()))


def _load(path: str | Path) -> Document:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return Document(text=text, metadata={"source": str(file_path.resolve())})


def load_txt(path: str | Path) -> Document:
    """读取一个 UTF-8 txt 文件，并记录文件来源。"""

    return _load(path)


def load_markdown(path: str | Path) -> Document:
    """读取一个 UTF-8 Markdown 文件，并记录文件来源。"""

    return _load(path)
