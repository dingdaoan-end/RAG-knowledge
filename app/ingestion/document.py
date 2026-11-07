from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """一段文本以及描述这段文本的元数据。"""

    text: str
    metadata: dict[str, Any]
