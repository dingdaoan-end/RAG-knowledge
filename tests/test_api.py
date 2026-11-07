import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.generation.llm import BaseLLM, MockLLM
from app.ingestion.document import Document
from app.pipeline import RAGPipeline
from app.retrieval.retriever import SearchResult


class MemoryRetriever:
    def search(self, question):
        return [SearchResult(Document("RAG 是检索增强生成。", {"source": "rag.txt", "chunk_id": 0}), 0.9, 1)]


@pytest.fixture
def client():
    from main import create_app

    with TestClient(create_app(RAGPipeline(MemoryRetriever(), MockLLM()))) as client:
        yield client


def parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


def test_chat_returns_answer(client):
    response = client.post("/chat", json={"question": "RAG 是什么？"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": MockLLM().generate("prompt"), "status": "answered",
        "sources": [{"document_id": "source_b8637f6a7b790a8baf88e5f8b4856a40d18269bd4f03e0f3fee8907414a5216f",
                     "source": "rag.txt", "chunk_id": 0, "content": "RAG 是检索增强生成。", "rank": 1, "score": 0.9}],
    }


def test_chat_stream_returns_incremental_tokens_and_done(client):
    response = client.post("/chat/stream", json={"question": "RAG 是什么？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    tokens = [payload["token"] for event, payload in events if event == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == MockLLM().generate("prompt")
    assert events[-1] == ("done", {})


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "  \n"}, {"question": 123}])
def test_invalid_questions_are_rejected(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_pipeline_stream_keeps_prompt_and_is_lazy():
    class RecordingLLM(BaseLLM):
        def stream_generate(self, prompt):
            assert "RAG 是检索增强生成。" in prompt
            assert "RAG 是什么？" in prompt
            self.steps.append(1)
            yield "RAG"
            self.steps.append(2)
            yield " 是检索增强生成。"

    llm = RecordingLLM()
    llm.steps = []
    iterator = RAGPipeline(MemoryRetriever(), llm).stream_ask("RAG 是什么？")
    assert llm.steps == []
    assert next(iterator) == "RAG"
    assert llm.steps == [1]
    assert "".join(iterator) == " 是检索增强生成。"


def test_errors_before_streaming_return_502_without_secret():
    from main import create_app

    class FailedLLM(BaseLLM):
        def generate(self, prompt):
            raise RuntimeError("test-secret-key")

        def stream_generate(self, prompt):
            raise RuntimeError("test-secret-key")
            yield "unreachable"

    with TestClient(create_app(RAGPipeline(MemoryRetriever(), FailedLLM()))) as client:
        for path in ["/chat", "/chat/stream"]:
            response = client.post(path, json={"question": "test"})
            assert response.status_code == 502
            assert "test-secret-key" not in response.text


def test_errors_after_streaming_are_reported_as_error_event():
    from main import create_app

    class FailedLLM(BaseLLM):
        def stream_generate(self, prompt):
            yield "部分答案"
            raise RuntimeError("test-secret-key")

    with TestClient(create_app(RAGPipeline(MemoryRetriever(), FailedLLM()))) as client:
        response = client.post("/chat/stream", json={"question": "test"})
        events = parse_events(response.text)
        assert events[0][0] == "sources"
        assert events[1] == ("token", {"token": "部分答案"})
        assert events[-1][0] == "error"
        assert not any(event == "done" for event, _ in events)
        assert "test-secret-key" not in response.text


@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
def test_disconnect_closes_upstream_generator(failed_message):
    from app.api.routes import ChatRequest, chat_stream
    from starlette.requests import ClientDisconnect

    class TrackingLLM(BaseLLM):
        closed = False

        def stream_generate(self, prompt):
            try:
                yield "第一段"
                yield "第二段"
            finally:
                self.closed = True

    llm = TrackingLLM()
    response = chat_stream(ChatRequest(question="test"), RAGPipeline(MemoryRetriever(), llm))
    assert not llm.closed

    async def send(message):
        if message["type"] == failed_message:
            raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    async def invoke():
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        # 在事件循环关闭前检查，避免 asyncio 自动清理掩盖连接泄漏。
        assert llm.closed

    asyncio.run(invoke())


def test_asgi23_disconnect_cancellation_closes_upstream():
    from app.api.routes import ChatRequest, chat_stream

    class TrackingLLM(BaseLLM):
        closed = False

        def stream_generate(self, prompt):
            try:
                yield "第一段"
                yield "第二段"
            finally:
                self.closed = True

    async def invoke():
        llm = TrackingLLM()
        response = chat_stream(ChatRequest(question="test"), RAGPipeline(MemoryRetriever(), llm))
        first_body = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                first_body.set()
                await asyncio.Event().wait()

        async def receive():
            await first_body.wait()
            return {"type": "http.disconnect"}

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        assert llm.closed

    asyncio.run(invoke())


@pytest.mark.parametrize("tokens", [[], [""], [" \n", "\t"]])
def test_empty_custom_stream_returns_502_and_closes_generator(tokens):
    from main import create_app

    class EmptyLLM(BaseLLM):
        closed = False

        def stream_generate(self, prompt):
            try:
                yield from tokens
            finally:
                self.closed = True

    llm = EmptyLLM()
    with TestClient(create_app(RAGPipeline(MemoryRetriever(), llm))) as client:
        response = client.post("/chat/stream", json={"question": "问题"})
        assert response.status_code == 502
        assert "event: done" not in response.text
        assert llm.closed
