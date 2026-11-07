import pytest

from app.generation.llm import BaseLLM
from app.ingestion.document import Document
from app.pipeline import RAGPipeline
from app.retrieval.retriever import SearchResult


def test_mock_reranker_preserves_document_order_and_metadata():
    from app.retrieval.reranker import MockReranker, Reranker

    documents = [Document("RAG", {"source": "a.txt"}), Document("BM25", {"source": "b.txt"})]
    output = MockReranker().rerank("query", documents)
    assert output == documents
    assert output is not documents
    assert output[0] is documents[0]
    assert MockReranker().rerank("query", []) == []
    with pytest.raises(NotImplementedError):
        Reranker().rerank("query", documents)


@pytest.mark.parametrize("streaming", [False, True])
def test_pipeline_rewrites_then_reranks_before_final_top_k(streaming):
    from app.query.rewrite import QueryRewriter
    from app.retrieval.reranker import Reranker

    calls = []
    documents = [Document("第一条", {"source": "a.txt"}), Document("第二条", {"source": "b.txt"})]
    original_results = [SearchResult(documents[0], 0.9, 1), SearchResult(documents[1], 0.4, 2)]

    class Rewriter(QueryRewriter):
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "改写后的检索问题"

    class Retriever:
        def search(self, query):
            calls.append(("retrieve", query))
            return original_results

    class ReverseReranker(Reranker):
        def rerank(self, query, documents):
            calls.append(("rerank", query))
            return list(reversed(documents))

    class RecordingLLM(BaseLLM):
        def generate(self, prompt):
            calls.append(("generate", prompt))
            return "答案"

        def stream_generate(self, prompt):
            yield self.generate(prompt)

    pipeline = RAGPipeline(Retriever(), RecordingLLM(), query_rewriter=Rewriter(), reranker=ReverseReranker(), top_k=1)
    answer = "".join(pipeline.stream_ask("原问题")) if streaming else pipeline.ask("原问题")
    assert answer == "答案"
    assert calls[:3] == [("rewrite", "原问题"), ("retrieve", "改写后的检索问题"), ("rerank", "原问题")]
    prompt = calls[-1][1]
    assert "source: b.txt" in prompt and "source: a.txt" not in prompt
    assert "rank: 1" in prompt and "score: 0.4" in prompt
    assert "Question:\n\n原问题" in prompt
    assert [result.rank for result in original_results] == [1, 2]


def test_reranker_cannot_insert_foreign_documents():
    from app.retrieval.reranker import Reranker

    class Retriever:
        def search(self, query):
            return [SearchResult(Document("候选", {}), 0.9, 1)]

    class InvalidReranker(Reranker):
        def rerank(self, query, documents):
            return [Document("不存在的资料", {})]

    pipeline = RAGPipeline(Retriever(), BaseLLM(), reranker=InvalidReranker())
    with pytest.raises(ValueError, match="候选"):
        pipeline.ask("query")
