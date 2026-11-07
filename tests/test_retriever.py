import numpy as np
import pytest

from app.ingestion.document import Document
from app.retrieval.retriever import FaissRetriever


@pytest.fixture
def retriever():
    documents = [
        Document("向量检索", {"source": "rag.txt", "chunk_id": 0}),
        Document("语义检索", {"source": "rag.txt", "chunk_id": 1}),
        Document("公园散步", {"source": "park.txt", "chunk_id": 0}),
    ]
    vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype="float32")
    return FaissRetriever(documents, vectors)


def test_results_include_document_score_rank_and_metadata(retriever):
    results = retriever.search(np.array([[1, 0]], dtype="float32"), top_k=3)
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == pytest.approx([1, 0.8, 0])
    assert results[0].document.text == "向量检索"
    assert results[0].document.metadata == {"source": "rag.txt", "chunk_id": 0}


@pytest.mark.parametrize("top_k, count", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_top_k_count(retriever, top_k, count):
    assert len(retriever.search(np.array([[1, 0]]), top_k)) == count


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "3", None])
def test_invalid_top_k_is_rejected(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.search(np.array([[1, 0]]), top_k)


def test_empty_index_returns_empty_results():
    retriever = FaissRetriever([], np.empty((0, 2), dtype="float32"))
    assert retriever.search(np.array([[1, 0]]), top_k=5) == []
    with pytest.raises(ValueError):
        retriever.search(np.array([[1, 0]]), top_k=0)


@pytest.mark.parametrize("vectors", [np.ones(2), np.ones((1, 2)), np.ones((3, 0)), np.full((3, 2), np.nan)])
def test_invalid_document_embeddings_are_rejected(vectors):
    with pytest.raises(ValueError):
        FaissRetriever([Document("text", {}) for _ in range(3)], vectors)


@pytest.mark.parametrize("query", [np.ones((2, 2)), np.ones((1, 3)), np.full((1, 2), np.inf)])
def test_invalid_query_embeddings_are_rejected(retriever, query):
    with pytest.raises(ValueError):
        retriever.search(query)


def test_one_dimensional_query_is_accepted(retriever):
    results = retriever.search(np.array([1, 0]), top_k=1)
    assert results[0].document.text == "向量检索"


def test_documents_match_search_results_and_preserve_chunks_compatibility():
    documents = [Document("RAG", {"source": "rag.txt", "chunk_id": 0})]
    retriever = FaissRetriever(chunks=documents, embeddings=np.array([[1, 0]], dtype="float32"))
    result = retriever.search(np.array([1, 0]))[0]
    assert retriever.documents[0] is documents[0]
    assert result.document is retriever.documents[0]
    assert retriever.chunks is retriever.documents
    assert result.score == 1
