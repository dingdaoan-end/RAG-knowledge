from dataclasses import dataclass

import faiss
import numpy as np

from app.ingestion.document import Document


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class FaissRetriever:
    """精确内积检索；输入向量应由 Embedder 做 L2 归一化。"""

    def __init__(self, chunks: list[Document], embeddings: np.ndarray) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError("embeddings 必须是 (文档数量, 向量维度) 的二维数组")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("文档数量与向量数量必须一致")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings 不能包含 NaN 或无穷值")
        self.documents = list(chunks)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        if self.documents:
            self.index.add(embeddings)

    @property
    def chunks(self) -> list[Document]:
        """旧属性的兼容入口；检索器内部统一使用 documents。"""
        return self.documents

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[SearchResult]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape != (1, self.index.d):
            raise ValueError("查询必须是一个向量，且维度与索引一致")
        if not np.isfinite(query_embedding).all():
            raise ValueError("查询向量不能包含 NaN 或无穷值")
        if not self.documents:
            return []

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        return [
            SearchResult(self.documents[index], float(score), rank)
            for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1)
        ]
