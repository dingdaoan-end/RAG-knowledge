import math
import re
from dataclasses import dataclass

import jieba
import numpy as np
from rank_bm25 import BM25L

from app.ingestion.document import Document
from .retriever import SearchResult


def tokenize(text: str) -> list[str]:
    """中文用 jieba；英文统一小写，保留 ERR-42 一类完整标识符。"""
    tokens = []
    for part in re.findall(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)*|[\u4e00-\u9fff]+", text.lower()):
        if "\u4e00" <= part[0] <= "\u9fff":
            tokens.extend(jieba.lcut(part, HMM=False))
        else:
            tokens.append(part)
    return tokens


def _validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数")


def normalize_scores(scores: np.ndarray | list[float]) -> np.ndarray:
    """对一侧实际结果做 Min-Max；空结果保持空，单结果/等分结果设为 1。"""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores 必须是一维分数列表")
    if not np.isfinite(scores).all():
        raise ValueError("检索分数不能包含 NaN 或无穷值")
    if scores.size == 0:
        return scores.copy()
    minimum, maximum = scores.min(), scores.max()
    if maximum == minimum:
        # 分母为零时无法计算 Min-Max；保留已召回结果的同等贡献。
        return np.ones_like(scores)
    return (scores - minimum) / (maximum - minimum)


@dataclass
class HybridSearchResult:
    document: Document
    vector_score: float | None
    bm25_score: float | None
    normalized_vector_score: float
    normalized_bm25_score: float
    final_score: float
    rank: int

    @property
    def score(self) -> float:
        """兼容原 Context Builder、评估脚本等对 result.score 的读取。"""
        return self.final_score


class BM25Retriever:
    """用 rank-bm25 的 BM25L 进行词项检索，返回原始 BM25 分数。"""

    def __init__(self, documents: list[Document], top_k: int = 3) -> None:
        _validate_top_k(top_k)
        self.documents = list(documents)
        self.top_k = top_k
        corpus = [tokenize(document.text) for document in self.documents]
        # 空语料或全部没有有效词项时不创建索引，避免除零。
        self.index = BM25L(corpus) if any(corpus) else None

    def get_scores(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query 不能为空")
        if self.index is None:
            return np.zeros(len(self.documents))
        return self.index.get_scores(tokenize(query))

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        count = self.top_k if top_k is None else top_k
        _validate_top_k(count)
        scores = self.get_scores(query)
        # BM25L 分数为 0 表示没有匹配词项，不返回任意不相关文档凑数。
        indices = [index for index in np.argsort(-scores, kind="stable") if scores[index] > 0][:count]
        return [SearchResult(self.documents[index], float(scores[index]), rank) for rank, index in enumerate(indices, 1)]


class HybridRetriever:
    """先按同一文档对齐两路分数，再归一化、加权、排序。"""

    def __init__(self, vector_retriever, bm25_retriever: BM25Retriever, alpha: float = 0.7, top_k: int = 3) -> None:
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha 必须是 [0, 1] 内的有限数字")
        _validate_top_k(top_k)
        vector_documents = list(vector_retriever.documents)
        bm25_documents = list(bm25_retriever.documents)
        if vector_documents and bm25_documents and (
            len(vector_documents) != len(bm25_documents) or any(
                left is not right for left, right in zip(vector_documents, bm25_documents)
            )
        ):
            raise ValueError("两路索引必须使用同一组 Document 对象，且顺序一致")
        # 一侧索引为空时，也能使用另一侧的语料。
        self.documents = vector_documents or bm25_documents
        self.document_indices = {id(document): index for index, document in enumerate(self.documents)}
        if len(self.document_indices) != len(self.documents):
            raise ValueError("文档对象不能重复；内容相同的 chunk 应各自创建 Document")
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.top_k = top_k

    def _score_maps(self, results: list[SearchResult]) -> tuple[dict, dict]:
        ids = [id(result.document) for result in results]
        if len(set(ids)) != len(ids) or any(document_id not in self.document_indices for document_id in ids):
            raise ValueError("单路结果必须是不重复的索引内 Document 对象")
        raw_scores = [float(result.score) for result in results]
        normalized = normalize_scores(raw_scores)
        return dict(zip(ids, raw_scores)), dict(zip(ids, normalized.tolist()))

    def search(self, query: str, top_k: int | None = None) -> list[HybridSearchResult]:
        count = self.top_k if top_k is None else top_k
        _validate_top_k(count)
        if not query.strip():
            raise ValueError("query 不能为空")
        if not self.documents:
            return []
        # 获取实际结果；即使一侧权重为 0，也保留其原始分数供调试。
        vector_results = self.vector_retriever.search(query, top_k=len(self.documents))
        bm25_results = self.bm25_retriever.search(query, top_k=len(self.documents))
        vector_raw, vector_normalized = self._score_maps(vector_results)
        bm25_raw, bm25_normalized = self._score_maps(bm25_results)
        candidate_ids = set()
        if self.alpha > 0:
            candidate_ids.update(vector_raw)
        if self.alpha < 1:
            candidate_ids.update(bm25_raw)

        results = []
        for document in self.documents:
            document_id = id(document)
            if document_id not in candidate_ids:
                continue
            vector_score = vector_normalized.get(document_id, 0.0)
            bm25_score = bm25_normalized.get(document_id, 0.0)
            final_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            results.append(HybridSearchResult(
                document=document,
                vector_score=vector_raw.get(document_id),
                bm25_score=bm25_raw.get(document_id),
                normalized_vector_score=vector_score,
                normalized_bm25_score=bm25_score,
                final_score=final_score,
                rank=0,
            ))
        # 稳定排序：同分时保持语料顺序；只给实际召回的候选分配最终 rank。
        results = sorted(results, key=lambda result: -result.final_score)[:count]
        for rank, result in enumerate(results, 1):
            result.rank = rank
        return results
