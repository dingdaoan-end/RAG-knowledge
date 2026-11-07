import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.document import Document


class TextEmbedder:
    """使用 sentence-transformers 生成归一化文本向量。"""

    def __init__(
        self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype="float32")
        # encode 内部按 batch_size 分批，输出仍保持输入顺序。
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(embeddings, dtype="float32")

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """返回 (N, d)；metadata 不参与模型编码。"""
        return self._encode([document.text for document in documents])

    def embed_query(self, query: str) -> np.ndarray:
        """返回 (1, d)，可直接传给 FAISS search。"""
        if not query.strip():
            raise ValueError("查询不能为空或只有空白字符")
        return self._encode([query])
