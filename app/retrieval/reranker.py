from dataclasses import dataclass

import numpy as np

from app.ingestion.document import Document
from .hybrid import HybridSearchResult
from .retriever import SearchResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


@dataclass
class RerankResult:
    document: Document
    rerank_score: float
    rank: int
    vector_score: float | None = None
    bm25_score: float | None = None
    normalized_vector_score: float | None = None
    normalized_bm25_score: float | None = None
    final_score: float | None = None
    retrieval_score: float | None = None
    retrieval_rank: int | None = None

    @property
    def score(self) -> float:
        """Context Builder 使用当前阶段的重排分数；原分数单独保留。"""
        return self.rerank_score

    @property
    def hybrid_score(self) -> float | None:
        return self.final_score


class Reranker:
    """保留已有 Document 接口；结果型实现声明 accepts_results=True。"""

    accepts_results = False

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        raise NotImplementedError("子类必须实现 rerank(query, documents)")


class MockReranker(Reranker):
    """保留候选顺序，仅验证重排接口，不声称提高相关性。"""

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        return list(documents)


class CrossEncoderReranker(Reranker):
    """联合编码 query 与已召回候选；不持有或搜索知识库。"""

    accepts_results = True

    def __init__(
        self, model_name: str = DEFAULT_RERANKER_MODEL, batch_size: int = 16,
        max_length: int = 512, device: str | None = None,
        local_files_only: bool = False, *, scorer=None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name 不能为空")
        for name, value in [("batch_size", batch_size), ("max_length", max_length)]:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.local_files_only = local_files_only
        self._scorer = scorer

    def _get_scorer(self):
        if self._scorer is None:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            # 延迟加载：空候选无需模型；注入 fake scorer 的单元测试无需下载。
            self._scorer = CrossEncoder(
                self.model_name, max_length=self.max_length, device=self.device,
                local_files_only=self.local_files_only,
                default_activation_function=Identity(),
            )
        return self._scorer

    def rerank(
        self, query: str, hybrid_results: list[HybridSearchResult | SearchResult],
        top_k: int = 3,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if not hybrid_results:
            return []
        document_ids = [id(result.document) for result in hybrid_results]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("候选结果不能包含重复 Document 对象")

        pairs = [(query, result.document.text) for result in hybrid_results]
        predicted = self._get_scorer().predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        if np.iscomplexobj(predicted):
            raise ValueError("Cross-Encoder 分数必须是实数")
        try:
            scores = np.asarray(predicted, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Cross-Encoder 分数必须是实数") from error
        # 单标签 Cross-Encoder 应为每个 pair 返回一个有限实数。
        if scores.shape == (len(pairs), 1):
            scores = scores[:, 0]
        if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
            raise ValueError("Cross-Encoder 必须为每个候选返回一个有限分数")

        # 稳定排序：同分时保留 Hybrid 顺序；先评分所有候选，再截取 Top-K。
        order = sorted(range(len(pairs)), key=lambda index: -scores[index])[:top_k]
        results = []
        for rank, index in enumerate(order, 1):
            original = hybrid_results[index]
            results.append(RerankResult(
                document=original.document, rerank_score=float(scores[index]), rank=rank,
                vector_score=getattr(original, "vector_score", None),
                bm25_score=getattr(original, "bm25_score", None),
                normalized_vector_score=getattr(original, "normalized_vector_score", None),
                normalized_bm25_score=getattr(original, "normalized_bm25_score", None),
                final_score=getattr(original, "final_score", None),
                retrieval_score=original.score, retrieval_rank=original.rank,
            ))
        return results
