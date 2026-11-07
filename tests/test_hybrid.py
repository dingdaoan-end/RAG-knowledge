import numpy as np
import pytest

from app.ingestion.document import Document
from app.retrieval.retriever import FaissRetriever, SearchResult
from app.retrieval.text_retriever import TextRetriever


def test_bm25_matches_chinese_and_exact_identifiers():
    from app.retrieval.hybrid import BM25Retriever

    documents = [
        Document("足球比赛与体育运动", {"source": "sports.txt"}),
        Document("向量检索错误 ERR-42 的处理办法", {"source": "rag.txt", "chunk_id": 1}),
    ]
    retriever = BM25Retriever(documents)
    for query in ["向量检索", "err-42"]:
        results = retriever.search(query, top_k=10)
        assert results[0].document is documents[1]
        assert results[0].score > 0
        assert results[0].rank == 1
        assert results[0].document.metadata == {"source": "rag.txt", "chunk_id": 1}


def test_bm25_empty_documents_and_unknown_terms():
    from app.retrieval.hybrid import BM25Retriever

    assert BM25Retriever([]).search("RAG") == []
    assert BM25Retriever([Document(" \n", {})]).search("RAG") == []
    assert BM25Retriever([Document("RAG", {})]).search("unseenword") == []
    documents = [Document("", {}), Document("RAG retrieval", {})]
    assert BM25Retriever(documents).search("RAG")[0].document is documents[1]


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3"])
def test_bm25_rejects_invalid_top_k(top_k):
    from app.retrieval.hybrid import BM25Retriever

    with pytest.raises(ValueError):
        BM25Retriever([]).search("RAG", top_k=top_k)


class FixedVectorRetriever:
    def __init__(self, documents, scores):
        self.documents = documents
        self.scores = scores
        self.requested_k = []

    def search(self, query, top_k=None):
        self.requested_k.append(top_k)
        order = sorted(range(len(self.documents)), key=lambda index: -self.scores[index])
        return [SearchResult(self.documents[index], self.scores[index], rank) for rank, index in enumerate(order[:top_k], 1)]


class FixedBM25Retriever:
    def __init__(self, documents, scores):
        self.documents = documents
        self.scores = np.array(scores, dtype=float)

    def get_scores(self, query):
        return self.scores.copy()

    def search(self, query, top_k=None):
        indices = [index for index in np.argsort(-self.scores, kind="stable") if self.scores[index] > 0]
        return [SearchResult(self.documents[index], float(self.scores[index]), rank) for rank, index in enumerate(indices[:top_k], 1)]


def make_hybrid(alpha=0.5, vector_scores=(0.2, 0.8, 0.5), bm25_scores=(20, 0, 10)):
    from app.retrieval.hybrid import HybridRetriever

    documents = [Document(str(index), {"source": "same.txt", "chunk_id": index}) for index in range(3)]
    vector = FixedVectorRetriever(documents, vector_scores)
    bm25 = FixedBM25Retriever(documents, bm25_scores)
    return HybridRetriever(vector, bm25, alpha=alpha), vector


def test_hybrid_normalizes_scores_before_weighted_fusion():
    retriever, vector = make_hybrid(alpha=0.75)
    results = retriever.search("query", top_k=3)
    assert [result.document.text for result in results] == ["1", "2", "0"]
    # BM25 实际只返回 20、10；分别归一化为 1、0，未返回侧贡献为 0。
    assert [result.score for result in results] == pytest.approx([0.75, 0.375, 0.25])
    assert results[0].final_score == results[0].score
    assert results[0].vector_score == 0.8
    assert results[0].bm25_score is None
    assert results[0].normalized_vector_score == 1
    assert results[0].normalized_bm25_score == 0
    assert [result.rank for result in results] == [1, 2, 3]
    # 两路都覆盖全语料，避免提前 Top-K 截断改变融合排名。
    assert vector.requested_k == [3]


@pytest.mark.parametrize("alpha, first", [(0, "0"), (1, "1")])
def test_alpha_endpoints_select_each_retrieval_branch(alpha, first):
    retriever, _ = make_hybrid(alpha=alpha)
    assert retriever.search("query", top_k=1)[0].document.text == first


def test_flat_scores_do_not_produce_nan_or_division_by_zero():
    retriever, _ = make_hybrid(vector_scores=(0.5, 0.5, 0.5), bm25_scores=(0, 0, 0))
    results = retriever.search("query", top_k=10)
    assert len(results) == 3
    assert [result.score for result in results] == [0.5, 0.5, 0.5]
    assert [result.normalized_vector_score for result in results] == [1, 1, 1]
    assert [result.document.metadata["chunk_id"] for result in results] == [0, 1, 2]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_hybrid_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        make_hybrid(alpha=alpha)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_hybrid_rejects_invalid_top_k(top_k):
    retriever, _ = make_hybrid()
    with pytest.raises(ValueError):
        retriever.search("query", top_k=top_k)


def test_hybrid_requires_same_document_objects_in_both_indexes():
    from app.retrieval.hybrid import HybridRetriever

    first = [Document("RAG", {})]
    second = [Document("RAG", {})]
    with pytest.raises(ValueError):
        HybridRetriever(FixedVectorRetriever(first, [1]), FixedBM25Retriever(second, [1]))


def test_hybrid_preserves_distinct_chunks_with_equal_text():
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever

    documents = [Document("RAG", {"chunk_id": 0}), Document("RAG", {"chunk_id": 1})]

    class Embedder:
        def embed_query(self, query):
            return np.array([[1, 0]], dtype="float32")

    vector = TextRetriever(FaissRetriever(documents, np.array([[1, 0], [0, 1]], dtype="float32")), Embedder())
    results = HybridRetriever(vector, BM25Retriever(documents)).search("RAG", top_k=2)
    assert [result.document.metadata["chunk_id"] for result in results] == [0, 1]


def test_hybrid_empty_index():
    from app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(FixedVectorRetriever([], []), FixedBM25Retriever([], []))
    assert retriever.search("query") == []


@pytest.mark.parametrize("scores, expected", [
    ([], []), ([0.85], [1]), ([5, 5, 5], [1, 1, 1]),
    ([0, 0], [1, 1]), ([-0.8, -0.5, -0.2], [0, 0.5, 1]),
    ([10, 20, 30], [0, 0.5, 1]),
])
def test_min_max_normalization(scores, expected):
    from app.retrieval.hybrid import normalize_scores

    assert normalize_scores(scores).tolist() == pytest.approx(expected)


class ReturnedResultsRetriever:
    def __init__(self, documents, results):
        self.documents = documents
        self.results = results

    def search(self, query, top_k=None):
        return self.results[:top_k]


def test_no_results_does_not_return_unretrieved_documents():
    from app.retrieval.hybrid import HybridRetriever

    documents = [Document("未召回的文档", {})]
    empty = ReturnedResultsRetriever(documents, [])
    assert HybridRetriever(empty, empty).search("query") == []


@pytest.mark.parametrize("side, expected", [("vector", 0.7), ("bm25", 0.3)])
def test_single_result_and_empty_other_branch(side, expected):
    from app.retrieval.hybrid import HybridRetriever

    document = Document("RAG", {"source": "rag.txt", "chunk_id": 7})
    documents = [document]
    hit = ReturnedResultsRetriever(documents, [SearchResult(document, 0.85, 1)])
    empty = ReturnedResultsRetriever(documents, [])
    vector, bm25 = (hit, empty) if side == "vector" else (empty, hit)
    result = HybridRetriever(vector, bm25).search("query")[0]
    assert result.final_score == pytest.approx(expected)
    assert result.document is document
    assert result.rank == 1
    assert getattr(result, f"{side}_score") == 0.85
    missing_side = "bm25" if side == "vector" else "vector"
    assert getattr(result, f"{missing_side}_score") is None
    assert getattr(result, f"normalized_{missing_side}_score") == 0


@pytest.mark.parametrize("empty_side", ["vector", "bm25"])
def test_one_index_can_have_no_documents(empty_side):
    from app.retrieval.hybrid import HybridRetriever

    document = Document("RAG", {})
    hit = ReturnedResultsRetriever([document], [SearchResult(document, 2, 1)])
    empty = ReturnedResultsRetriever([], [])
    vector, bm25 = (empty, hit) if empty_side == "vector" else (hit, empty)
    assert HybridRetriever(vector, bm25).search("query")[0].document is document


@pytest.mark.parametrize("alpha, expected_document", [(1, "vector"), (0, "bm25")])
def test_zero_weight_branch_does_not_add_its_only_candidates(alpha, expected_document):
    from app.retrieval.hybrid import HybridRetriever

    documents = [Document("vector", {}), Document("bm25", {})]
    vector = ReturnedResultsRetriever(documents, [SearchResult(documents[0], -0.5, 1)])
    bm25 = ReturnedResultsRetriever(documents, [SearchResult(documents[1], 9, 1)])
    results = HybridRetriever(vector, bm25, alpha=alpha).search("query")
    assert [result.document.text for result in results] == [expected_document]
    assert results[0].final_score == 1


def test_negative_vector_scores_are_normalized_without_missing_zeros():
    from app.retrieval.hybrid import HybridRetriever

    documents = [Document("a", {}), Document("b", {}), Document("c", {})]
    vector = ReturnedResultsRetriever(documents, [SearchResult(documents[0], -0.2, 1), SearchResult(documents[1], -0.8, 2)])
    bm25 = ReturnedResultsRetriever(documents, [SearchResult(documents[2], 100, 1)])
    results = HybridRetriever(vector, bm25, alpha=0.7).search("query")
    assert [result.document.text for result in results] == ["a", "c", "b"]
    assert [result.final_score for result in results] == pytest.approx([0.7, 0.3, 0])


def test_debug_fields_are_serializable_and_work_with_context_and_reranking():
    from dataclasses import asdict, replace
    from app.generation.context_builder import build_context

    result = make_hybrid(alpha=0.75)[0].search("query")[0]
    payload = asdict(result)
    assert set(payload) == {
        "document", "vector_score", "bm25_score", "normalized_vector_score",
        "normalized_bm25_score", "final_score", "rank",
    }
    assert result.score == result.final_score
    assert "score: 0.75" in build_context([result])
    # 构造字典格式时保留真实 Document；asdict 的 document 会递归转字典。
    assert "score: 0.75" in build_context([{**payload, "document": result.document}])
    assert replace(result, rank=2).final_score == 0.75


@pytest.mark.parametrize("alpha, expected_order, expected_scores", [
    (0.7, [0, 1], [0.7, 0.3]),
    (1, [0, 1], [1, 0]),
    (0, [1], [1]),
])
def test_real_faiss_and_bm25_fuse_without_confusing_source_local_chunk_ids(alpha, expected_order, expected_scores):
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever

    # 不同源文件的 chunk_id 可以相同，不能据此合并两个检索单元。
    documents = [
        Document("semantic answer", {"source": "first.txt", "chunk_id": 0}),
        Document("target target", {"source": "second.txt", "chunk_id": 0}),
    ]

    class Embedder:
        def embed_query(self, query):
            return np.array([[1, 0]], dtype="float32")

    faiss = FaissRetriever(documents, np.eye(2, dtype="float32"))
    vector = TextRetriever(faiss, Embedder())
    bm25 = BM25Retriever(documents)
    hybrid = HybridRetriever(vector, bm25, alpha=alpha)
    results = hybrid.search("target", top_k=10)

    assert vector.documents is faiss.documents
    assert [result.document for result in results] == [documents[index] for index in expected_order]
    assert [result.final_score for result in results] == pytest.approx(expected_scores)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert len(hybrid.search("target", top_k=1)) == 1
    assert hybrid.search("target", top_k=1)[0].document is documents[expected_order[0]]
    by_source = {result.document.metadata["source"]: result for result in results}
    lexical = by_source["second.txt"]
    assert lexical.vector_score == 0
    assert lexical.bm25_score == pytest.approx(bm25.search("target")[0].score)
    assert lexical.normalized_vector_score == 0
    assert lexical.normalized_bm25_score == 1
    if alpha > 0:
        semantic = by_source["first.txt"]
        assert semantic.vector_score == 1
        assert semantic.bm25_score is None
        assert semantic.normalized_vector_score == 1
        assert semantic.normalized_bm25_score == 0
