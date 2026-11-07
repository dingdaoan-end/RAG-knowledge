import json

import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app.generation.llm import BaseLLM
from app.ingestion.document import Document
from app.pipeline import RAGPipeline
from app.retrieval.retriever import FaissRetriever
from app.retrieval.text_retriever import TextRetriever


def traced_pipeline(*, empty=False, rerank_empty=False, failure=None):
    """真实 TextRetriever/FAISS；只替代模型推理和网络，记录调用顺序。"""
    calls = []
    documents = [] if empty else [Document("检索证据正文", {"source": "notes.txt", "chunk_id": 0})]

    class Embedder:
        def embed_query(self, question):
            calls.append("embedding")
            return np.array([[1, 0]], dtype="float32")

    class Retriever(TextRetriever):
        def search(self, question):
            calls.append("retriever")
            return super().search(question)

    class Rewriter:
        def rewrite(self, question):
            calls.append("rewrite")
            return question

    class Reranker:
        accepts_results = True

        def rerank(self, question, results, top_k):
            calls.append("reranker")
            if failure == "reranker":
                raise RuntimeError("reranker failed")
            return [] if rerank_empty else results[:top_k]

    class LLM(BaseLLM):
        def generate(self, prompt):
            calls.append("llm")
            assert "检索证据正文" in prompt
            if failure == "llm":
                raise APITimeoutError(request=httpx2.Request("POST", "https://example.test"))
            return "模型答案"

        def stream_generate(self, prompt):
            answer = self.generate(prompt)
            yield answer[:2]
            yield answer[2:]

    vectors = np.empty((0, 2), dtype="float32") if empty else np.array([[1, 0]], dtype="float32")
    retriever = Retriever(FaissRetriever(documents, vectors), Embedder())
    return RAGPipeline(retriever, LLM(), query_rewriter=Rewriter(), reranker=Reranker(), top_k=1), calls


@pytest.mark.parametrize("question", ["", " \t\n"])
@pytest.mark.parametrize("method", ["ask", "stream_ask", "ask_with_result", "stream_ask_with_result"])
def test_blank_questions_fail_before_any_component(question, method):
    pipeline, calls = traced_pipeline()
    with pytest.raises(ValueError, match="question 不能为空"):
        output = getattr(pipeline, method)(question)
        if method.startswith("stream"):
            list(output)
    assert calls == []


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_candidates_skip_reranker_and_llm_and_preserve_string_interface(streaming):
    pipeline, calls = traced_pipeline(empty=True)
    answer = "".join(pipeline.stream_ask("问题")) if streaming else pipeline.ask("问题")
    assert isinstance(answer, str)
    assert "没有检索到" in answer and "补充" in answer
    assert calls == ["rewrite", "retriever", "embedding"]


@pytest.mark.parametrize("empty, rerank_empty", [(True, False), (False, True)])
@pytest.mark.parametrize("streaming", [False, True])
def test_structured_no_evidence_covers_empty_retrieval_and_empty_reranking(empty, rerank_empty, streaming):
    pipeline, calls = traced_pipeline(empty=empty, rerank_empty=rerank_empty)
    if streaming:
        results = list(pipeline.stream_ask_with_result("问题"))
        assert len(results) == 1
        result = results[0]
    else:
        result = pipeline.ask_with_result("问题")
    assert result.status == "no_evidence"
    assert "没有检索到" in result.answer
    assert calls == ["rewrite", "retriever", "embedding"] + ([] if empty else ["reranker"])


@pytest.mark.parametrize("method", ["ask", "stream_ask", "ask_with_result", "stream_ask_with_result"])
def test_nonempty_candidates_keep_generation_and_component_order(method):
    pipeline, calls = traced_pipeline()
    output = getattr(pipeline, method)("问题")
    if method == "stream_ask":
        assert "".join(output) == "模型答案"
    elif method == "stream_ask_with_result":
        chunks = list(output)
        assert "".join(chunk.answer for chunk in chunks) == "模型答案"
        assert all(chunk.status == "answered" for chunk in chunks)
    elif method == "ask_with_result":
        assert output.answer == "模型答案" and output.status == "answered"
    else:
        assert output == "模型答案"
    assert calls == ["rewrite", "retriever", "embedding", "reranker", "llm"]


@pytest.mark.parametrize("failure, error_type", [("llm", APITimeoutError), ("reranker", RuntimeError)])
@pytest.mark.parametrize("streaming", [False, True])
def test_execution_failures_are_not_converted_to_no_evidence(failure, error_type, streaming):
    pipeline, calls = traced_pipeline(failure=failure)
    with pytest.raises(error_type):
        if streaming:
            list(pipeline.stream_ask("问题"))
        else:
            pipeline.ask_with_result("问题")
    assert calls == ["rewrite", "retriever", "embedding", "reranker"] + (["llm"] if failure == "llm" else [])


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_no_evidence_http_is_valid_request_with_explicit_status_and_no_llm(path):
    from main import create_app

    pipeline, calls = traced_pipeline(empty=True)
    with TestClient(create_app(pipeline)) as client:
        response = client.post(path, json={"question": "问题"})
    assert response.status_code == 200
    if path == "/chat":
        assert response.json()["status"] == "no_evidence"
        assert "没有检索到" in response.json()["answer"]
    else:
        blocks = response.text.strip().split("\n\n")
        names = [block.splitlines()[0].removeprefix("event: ") for block in blocks]
        assert names == ["sources", "token", "done"]
        assert json.loads(blocks[-1].splitlines()[1].removeprefix("data: ")) == {"status": "no_evidence"}
        assert "没有检索到" in response.text
    assert calls == ["rewrite", "retriever", "embedding"]


@pytest.mark.parametrize("failure", ["llm", "reranker"])
@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_http_execution_failure_remains_502(failure, path):
    from main import create_app

    pipeline, calls = traced_pipeline(failure=failure)
    with TestClient(create_app(pipeline)) as client:
        response = client.post(path, json={"question": "问题"})
    assert response.status_code == 502
    assert "event: done" not in response.text
    assert calls.count("llm") == (1 if failure == "llm" else 0)
