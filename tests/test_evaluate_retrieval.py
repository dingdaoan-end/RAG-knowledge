import numpy as np
import pytest

from app.ingestion.document import Document
from app.retrieval.retriever import FaissRetriever


def test_recall_counts_all_relevant_documents_not_just_a_hit():
    from evaluate_retrieval import recall_at_k

    assert recall_at_k(["a", "c", "b"], {"a", "b"}, 1) == 0.5
    assert recall_at_k(["a", "c", "b"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["a", "a"], {"a", "b"}, 3) == 0.5
    assert recall_at_k([], {"a"}, 5) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_evaluation_averages_query_recall():
    from evaluate_retrieval import evaluate_retrieval

    documents = [Document(text, {"id": text}) for text in ["a", "b", "c"]]
    retriever = FaissRetriever(documents, np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype="float32"))
    queries = np.array([[1, 0, 0], [0, 0, 1]], dtype="float32")
    metrics = evaluate_retrieval(retriever, queries, [{"a", "b"}, {"c"}])
    assert metrics == {1: 0.75, 3: 1.0, 5: 1.0}


def test_evaluation_handles_empty_index():
    from evaluate_retrieval import evaluate_retrieval

    retriever = FaissRetriever([], np.empty((0, 2), dtype="float32"))
    assert evaluate_retrieval(retriever, np.array([[1, 0]]), [{"a"}]) == {1: 0.0, 3: 0.0, 5: 0.0}


def test_evaluation_rejects_missing_or_mismatched_labels():
    from evaluate_retrieval import evaluate_retrieval

    retriever = FaissRetriever([], np.empty((0, 2), dtype="float32"))
    with pytest.raises(ValueError):
        evaluate_retrieval(retriever, np.empty((0, 2)), [])
    with pytest.raises(ValueError):
        evaluate_retrieval(retriever, np.array([[1, 0]]), [])
