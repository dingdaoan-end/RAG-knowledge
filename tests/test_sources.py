"""来源契约：只展示实际送入 Context Builder 的最终候选。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Barrier

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.routes import GENERATION_ERROR, router
from app.generation.context_builder import build_context
from app.generation.llm import BaseLLM
from app.generation.prompt import build_prompt
from app.ingestion.document import Document
from app.pipeline import NO_EVIDENCE_ANSWER, RAGPipeline, RAGResult
from app.retrieval.reranker import RerankResult
from app.retrieval.retriever import SearchResult


class FixedRetriever:
    def __init__(self, results):
        self.results, self.queries = results, []

    def search(self, question):
        self.queries.append(question)
        return self.results


class RecordingLLM(BaseLLM):
    def __init__(self, tokens=("答案",)):
        self.tokens, self.prompts = tokens, []
        self.closed = False

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "答案"

    def stream_generate(self, prompt):
        self.prompts.append(prompt)
        try:
            yield from self.tokens
        finally:
            self.closed = True


def evidence():
    return [
        SearchResult(Document("候选甲", {"document_id": "doc-a", "source": "a.md", "chunk_id": 0}), 0.9, 1),
        SearchResult(Document("候选乙", {"document_id": "doc-b", "eval_id": "eval-b", "source": "b.md", "chunk_id": 7, "page": 3}), 0.4, 2),
        SearchResult(Document("候选丙", {"eval_id": "eval-c", "source": "c.md", "chunk_id": 2}), 0.2, 3),
    ]


def client_for(pipeline):
    # 只装配路由，不读取配置或构建模型。
    app = FastAPI()
    app.state.pipeline = pipeline
    app.include_router(router)
    return TestClient(app)


def parse_events(text):
    return [(lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
            for lines in (block.splitlines() for block in text.strip().split("\n\n"))]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_sources_match_actual_context_after_reranking_and_final_top_k(legacy, streaming):
    candidates = evidence()
    retriever, llm = FixedRetriever(candidates), RecordingLLM(("答", "案"))

    class Reranker:
        accepts_results = not legacy

        def rerank(self, question, items, **kwargs):
            assert question == "原问题"
            if legacy:
                return [items[1], items[0], items[2]]
            assert kwargs == {"top_k": 1}
            return [RerankResult(items[1].document, 0.95, 1), RerankResult(items[0].document, 0.3, 2)]

    class Rewriter:
        def rewrite(self, question):
            assert question == "原问题"
            return "检索用问题"

    pipeline = RAGPipeline(retriever, llm, query_rewriter=Rewriter(), reranker=Reranker(), top_k=1)
    outcomes = list(pipeline.stream_ask_with_result("原问题")) if streaming else [pipeline.ask_with_result("原问题")]
    score = 0.4 if legacy else 0.95
    expected = [{"document_id": "doc-b", "source": "b.md", "chunk_id": 7,
                 "content": "候选乙", "rank": 1, "score": score, "page": 3}]
    assert all(outcome.sources == expected for outcome in outcomes)
    assert "".join(outcome.answer for outcome in outcomes) == "答案"
    selected = replace(candidates[1], rank=1) if legacy else RerankResult(candidates[1].document, 0.95, 1)
    assert llm.prompts == [build_prompt("原问题", build_context([selected]))]
    assert "候选甲" not in llm.prompts[0] and "候选丙" not in llm.prompts[0]
    assert retriever.queries == ["检索用问题"]
    assert [item.rank for item in candidates] == [1, 2, 3]


@pytest.mark.parametrize("path, display_name", [
    (r"C:\private\knowledge\notes.md", "notes.md"),
    ("/private/knowledge/notes.md", "notes.md"),
    (r"\\server\private\notes.md", "notes.md"),
])
def test_display_source_never_exposes_absolute_path_and_preserves_prompt(path, display_name):
    candidates = [SearchResult(Document("原正文", {"eval_id": "eval-id", "source": path, "chunk_id": 0}), 0.5, 4)]
    retriever, llm = FixedRetriever(candidates), RecordingLLM()
    result = RAGPipeline(retriever, llm).ask_with_result("问题")
    assert result.sources == [{"document_id": "eval-id", "source": display_name,
                               "chunk_id": 0, "content": "原正文", "rank": 4, "score": 0.5}]
    assert path in llm.prompts[0]  # 展示脱敏不改变实际上下文或 Prompt。
    assert candidates[0].document.metadata["source"] == path


def test_legacy_documents_have_stable_source_based_ids_and_missing_metadata_is_supported():
    documents = [Document("第一段", {"source": r"D:\data\legacy.md", "chunk_id": 0}),
                 Document("第二段", {"source": r"D:\data\legacy.md", "chunk_id": 1}), Document("无元数据", {})]
    candidates = [SearchResult(doc, 0.5, rank) for rank, doc in enumerate(documents, 1)]
    first = RAGPipeline(FixedRetriever(candidates), RecordingLLM()).ask_with_result("问题")
    second = RAGPipeline(FixedRetriever(candidates), RecordingLLM()).ask_with_result("问题")
    ids = [source["document_id"] for source in first.sources]
    assert all(isinstance(value, str) and value for value in ids)
    assert ids == [source["document_id"] for source in second.sources]
    assert ids[0] == ids[1] and ids[0] != ids[2]
    assert all("private" not in value and "\\" not in value and ":" not in value for value in ids)
    assert first.sources[2]["source"] == "未知来源"
    assert first.sources[2]["chunk_id"] is None


def test_result_default_sources_do_not_share_mutable_lists():
    first, second = RAGResult("a", "answered"), RAGResult("b", "answered")
    first.sources.append({"document_id": "only-a"})
    assert second.sources == []


@pytest.mark.parametrize("method", ["ask", "stream_ask"])
def test_string_interfaces_and_build_prompt_compatibility(method):
    candidates, llm = evidence(), RecordingLLM(("答", "案"))
    retriever = FixedRetriever(candidates)
    pipeline = RAGPipeline(retriever, llm, top_k=1)
    result = getattr(pipeline, method)("问题")
    answer = "".join(result) if method.startswith("stream") else result
    assert isinstance(answer, str) and answer == "答案"
    assert pipeline._build_prompt("问题") == llm.prompts[0] == build_prompt("问题", build_context(candidates[:1]))
    assert retriever.queries == ["问题", "问题"]  # 每次独立调用只有一次检索。


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rerank_empty", [False, True])
def test_no_evidence_has_empty_sources_and_never_calls_llm(streaming, rerank_empty):
    class EmptyReranker:
        accepts_results = True

        def rerank(self, question, items, top_k):
            return []

    llm = RecordingLLM()
    pipeline = RAGPipeline(FixedRetriever(evidence() if rerank_empty else []), llm,
                           reranker=EmptyReranker() if rerank_empty else None)
    outcomes = list(pipeline.stream_ask_with_result("问题")) if streaming else [pipeline.ask_with_result("问题")]
    assert len(outcomes) == 1
    assert outcomes[0].sources == []
    assert outcomes[0].answer == NO_EVIDENCE_ANSWER and outcomes[0].status == "no_evidence"
    assert llm.prompts == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("through_http", [False, True])
def test_concurrent_questions_keep_sources_request_local(streaming, through_http):
    barrier = Barrier(2)

    class Retriever:
        def search(self, question):
            return [SearchResult(Document(question, {"document_id": question, "source": question + ".md", "chunk_id": 0}), 1.0, 1)]

    class LLM(BaseLLM):
        def generate(self, prompt):
            barrier.wait(timeout=5)
            return "答案"

        def stream_generate(self, prompt):
            yield self.generate(prompt)
            barrier.wait(timeout=5)
            yield "结束"

    pipeline = RAGPipeline(Retriever(), LLM())

    def ask(question):
        if through_http:
            response = client.post("/chat/stream" if streaming else "/chat", json={"question": question})
            assert response.status_code == 200
            if streaming:
                events = parse_events(response.text)
                assert events[0][0] == "sources"
                return [events[0][1]["sources"]]
            return [response.json()["sources"]]
        chunks = list(pipeline.stream_ask_with_result(question)) if streaming else [pipeline.ask_with_result(question)]
        return [chunk.sources for chunk in chunks]

    with client_for(pipeline) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(ask, ["甲", "乙"]))
    for question, sources_per_chunk in zip(["甲", "乙"], outcomes):
        assert all(sources == [{"document_id": question, "source": question + ".md", "chunk_id": 0,
                                "content": question, "rank": 1, "score": 1.0}] for sources in sources_per_chunk)


def test_chat_returns_actual_sources_and_stream_emits_sources_before_preserved_prefix():
    llm = RecordingLLM((" \n", "\t", "答", "案"))
    retriever = FixedRetriever(evidence())
    with client_for(RAGPipeline(retriever, llm, top_k=1)) as client:
        ordinary = client.post("/chat", json={"question": "问题"})
        stream = client.post("/chat/stream", json={"question": "问题"})
    expected = [{"document_id": "doc-a", "source": "a.md", "chunk_id": 0,
                 "content": "候选甲", "rank": 1, "score": 0.9}]
    assert ordinary.status_code == stream.status_code == 200
    assert ordinary.json() == {"answer": "答案", "status": "answered", "sources": expected}
    assert parse_events(stream.text) == [("sources", {"sources": expected}),
                                         ("token", {"token": " \n\t答"}), ("token", {"token": "案"}), ("done", {})]
    assert retriever.queries == ["问题", "问题"]
    assert llm.closed


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_no_evidence_http_includes_empty_sources(path):
    llm = RecordingLLM()
    with client_for(RAGPipeline(FixedRetriever([]), llm)) as client:
        response = client.post(path, json={"question": "问题"})
    assert response.status_code == 200
    if path == "/chat":
        assert response.json() == {"answer": NO_EVIDENCE_ANSWER, "status": "no_evidence", "sources": []}
    else:
        assert parse_events(response.text) == [("sources", {"sources": []}),
                                              ("token", {"token": NO_EVIDENCE_ANSWER}), ("done", {"status": "no_evidence"})]
    assert llm.prompts == []


@pytest.mark.parametrize("prefix", [[], ["", " \n", "\t"]])
def test_failure_before_first_nonblank_token_is_502_without_sources_and_closes(prefix):
    class FailedLLM(RecordingLLM):
        def stream_generate(self, prompt):
            try:
                yield from prefix
                raise RuntimeError("private-secret")
            finally:
                self.closed = True

    llm = FailedLLM()
    with client_for(RAGPipeline(FixedRetriever(evidence()), llm)) as client:
        response = client.post("/chat/stream", json={"question": "问题"})
    assert response.status_code == 502
    assert response.json() == {"detail": GENERATION_ERROR}
    assert "event: sources" not in response.text and "private-secret" not in response.text
    assert llm.closed


def test_failure_after_sources_and_token_keeps_safe_error_without_done():
    class FailedLLM(RecordingLLM):
        def stream_generate(self, prompt):
            try:
                yield "部分答案"
                raise RuntimeError("private-secret")
            finally:
                self.closed = True

    llm = FailedLLM()
    with client_for(RAGPipeline(FixedRetriever(evidence()), llm, top_k=1)) as client:
        response = client.post("/chat/stream", json={"question": "问题"})
    assert response.status_code == 200
    events = parse_events(response.text)
    assert [event for event, _ in events] == ["sources", "token", "error"]
    assert events[0][1]["sources"][0]["document_id"] == "doc-a"
    assert events[1] == ("token", {"token": "部分答案"})
    assert events[2] == ("error", {"message": GENERATION_ERROR})
    assert "private-secret" not in response.text
    assert llm.closed
