import pytest


def test_query_rewriter_interface_requires_implementation():
    from app.query.rewrite import QueryRewriter

    with pytest.raises(NotImplementedError):
        QueryRewriter().rewrite("RAG")


def test_mock_rewriter_cleans_whitespace_without_inventing_information():
    from app.query.rewrite import MockQueryRewriter

    assert MockQueryRewriter().rewrite("  RAG 是什么？\n") == "RAG 是什么？"
    with pytest.raises(ValueError):
        MockQueryRewriter().rewrite(" \n")
