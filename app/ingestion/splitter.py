from copy import deepcopy

from .document import Document


RECURSIVE_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，")


def _validate_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if type(chunk_size) is not int or type(chunk_overlap) is not int:
        raise TypeError("chunk_size 和 chunk_overlap 必须是整数")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")


def _character_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step

    return chunks


def _to_documents(texts: list[str], document: Document) -> list[Document]:
    return [
        Document(
            text=text,
            metadata={**deepcopy(document.metadata), "chunk_id": chunk_id},
        )
        for chunk_id, text in enumerate(texts)
    ]


def character_split(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """使用固定大小的字符窗口切分文档。"""

    _validate_parameters(chunk_size, chunk_overlap)
    if not document.text.strip():
        return []
    texts = _character_chunks(document.text, chunk_size, chunk_overlap)
    return _to_documents(texts, document)


def paragraph_split(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """优先在段落边界结束窗口，找不到边界时按字符切分。"""

    _validate_parameters(chunk_size, chunk_overlap)
    text = document.text
    if not text.strip():
        return []

    texts: list[str] = []
    start = 0
    separator = "\n\n"

    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = maximum_end

        if maximum_end < len(text):
            search_start = start + chunk_overlap + 1
            boundary = text.rfind(separator, search_start, maximum_end)
            if boundary != -1:
                end = boundary + len(separator)

        texts.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap

    return _to_documents(texts, document)


def _split_and_keep_separator(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    return [
        part + separator if index < len(parts) - 1 else part
        for index, part in enumerate(parts)
        if part or index < len(parts) - 1
    ]


def _recursive_parts(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    for index, separator in enumerate(separators):
        if separator not in text:
            continue

        parts: list[str] = []
        for part in _split_and_keep_separator(text, separator):
            if len(part) <= chunk_size:
                parts.append(part)
            else:
                parts.extend(_recursive_parts(part, chunk_size, separators[index + 1 :]))
        return parts

    return _character_chunks(text, chunk_size, 0)


def _merge_parts(parts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) <= chunk_size:
            current += part
        else:
            chunks.append(current)
            available_overlap = max(0, chunk_size - len(part))
            overlap_length = min(chunk_overlap, available_overlap)
            overlap = current[-overlap_length:] if overlap_length else ""
            current = overlap + part

    if current:
        chunks.append(current)
    return chunks


def recursive_split(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """按分隔符优先级递归切分，再组合为不超过上限的 chunk。"""

    _validate_parameters(chunk_size, chunk_overlap)
    if not document.text.strip():
        return []

    parts = _recursive_parts(document.text, chunk_size, RECURSIVE_SEPARATORS)
    texts = _merge_parts(parts, chunk_size, chunk_overlap)
    return _to_documents(texts, document)
