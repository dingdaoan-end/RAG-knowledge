import json

import httpx2
import pytest
from openai import OpenAI


def response_payload():
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "status": "completed", "error": None, "incomplete_details": None,
        "instructions": None, "model": "test-model",
        "output": [{"id": "msg_test", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "知识库答案", "annotations": []}]}],
        "parallel_tool_calls": True, "temperature": 1, "tool_choice": "auto", "tools": [], "top_p": 1,
    }


def make_llm(monkeypatch, handler):
    import app.generation.llm as llm_module
    from config import Settings

    http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=http_client))
    return llm_module.OpenAILLM(settings=Settings(model_name="test-model", api_key="test-key", base_url="https://example.test/v1"))


def test_real_sdk_generate_sends_prompt_and_model(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=response_payload())

    llm = make_llm(monkeypatch, handler)
    try:
        assert llm.generate("资料与问题") == "知识库答案"
        assert str(requests[0].url) == "https://example.test/v1/responses"
        assert requests[0].headers["Authorization"] == "Bearer test-key"
        body = json.loads(requests[0].content)
        assert body["model"] == "test-model"
        assert body["input"] == "资料与问题"
    finally:
        llm.close()


def test_real_sdk_stream_yields_text_deltas(monkeypatch):
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        events = [
            {"type": "response.output_text.delta", "delta": "知识库", "sequence_number": 0, "item_id": "msg_test", "output_index": 0, "content_index": 0},
            {"type": "response.output_text.delta", "delta": "答案", "sequence_number": 1, "item_id": "msg_test", "output_index": 0, "content_index": 0},
            {"type": "response.completed", "sequence_number": 2, "response": response_payload()},
        ]
        content = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=content)

    llm = make_llm(monkeypatch, handler)
    try:
        assert list(llm.stream_generate("prompt")) == ["知识库", "答案"]
    finally:
        llm.close()


def test_stream_failure_is_not_silently_completed(monkeypatch):
    def handler(request):
        event = {"type": "error", "sequence_number": 0, "code": "server_error", "message": "upstream failure", "param": None}
        return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=f"data: {json.dumps(event)}\n\n")

    llm = make_llm(monkeypatch, handler)
    try:
        with pytest.raises(RuntimeError):
            list(llm.stream_generate("prompt"))
    finally:
        llm.close()


def test_missing_api_key_is_rejected(monkeypatch):
    from app.generation.llm import OpenAILLM

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API_KEY"):
        OpenAILLM()


def test_mock_stream_matches_normal_answer():
    from app.generation.llm import BaseLLM, MockLLM

    llm = MockLLM()
    iterator = llm.stream_generate("prompt")
    assert next(iterator) == llm.generate("prompt")[0]
    assert llm.generate("prompt")[0] + "".join(iterator) == llm.generate("prompt")
    with pytest.raises(NotImplementedError):
        BaseLLM().stream_generate("prompt")


class TrackedSSEBody(httpx2.SyncByteStream):
    """模拟网络响应，并检查真实 SDK 是否关闭响应体。"""

    def __init__(self, events):
        self.content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        self.closed = False

    def __iter__(self):
        yield self.content

    def close(self):
        self.closed = True


def streaming_events(deltas, terminal):
    events = [
        {"type": "response.output_text.delta", "delta": delta, "sequence_number": index,
         "item_id": "msg_test", "output_index": 0, "content_index": 0}
        for index, delta in enumerate(deltas)
    ]
    if terminal == "completed":
        payload = response_payload()
        payload["output"][0]["content"][0]["text"] = "".join(deltas)
        events.append({"type": "response.completed", "sequence_number": len(events), "response": payload})
    elif terminal == "error":
        events.append({"type": "error", "sequence_number": len(events), "code": "server_error",
                       "message": "private-upstream-error", "param": None})
    return events


@pytest.mark.parametrize("deltas, terminal, succeeds", [
    (["知识库", "答案"], "completed", True),
    ([" \n", "答案", " "], "completed", True),
    ([], "completed", False),
    ([""], "completed", False),
    ([" \n", "\t"], "completed", False),
    (["部分答案"], "error", False),
    (["部分答案"], None, False),
])
def test_stream_requires_completed_and_nonblank_text_and_closes_body(monkeypatch, deltas, terminal, succeeds):
    body = TrackedSSEBody(streaming_events(deltas, terminal))
    llm = make_llm(monkeypatch, lambda request: httpx2.Response(
        200, headers={"Content-Type": "text/event-stream"}, stream=body,
    ))
    try:
        if succeeds:
            assert "".join(llm.stream_generate("资料与问题")) == "".join(deltas)
        else:
            with pytest.raises(RuntimeError):
                list(llm.stream_generate("资料与问题"))
        assert body.closed
    finally:
        llm.close()


@pytest.mark.parametrize("deltas, terminal, expected_status, ending", [
    (["答案"], "completed", 200, "done"),
    ([" \n", "答案", " "], "completed", 200, "done"),
    ([], "completed", 502, None),
    ([" \n", "\t"], "completed", 502, None),
    ([], "error", 502, None),
    (["部分答案"], "error", 200, "error"),
    (["部分答案"], None, 200, "error"),
])
def test_real_provider_stream_http_outcome_and_cleanup(monkeypatch, deltas, terminal, expected_status, ending):
    from fastapi.testclient import TestClient
    from app.pipeline import RAGPipeline
    from app.ingestion.document import Document
    from app.retrieval.retriever import SearchResult
    from main import create_app

    class EvidenceRetriever:
        def search(self, question):
            # 非空候选才能实际调用 Provider，继续验证 Phase 1 的所有流式边界。
            return [SearchResult(Document("测试资料", {"source": "test.txt", "chunk_id": 0}), 0.9, 1)]

    body = TrackedSSEBody(streaming_events(deltas, terminal))
    llm = make_llm(monkeypatch, lambda request: httpx2.Response(
        200, headers={"Content-Type": "text/event-stream"}, stream=body,
    ))
    with TestClient(create_app(RAGPipeline(EvidenceRetriever(), llm))) as client:
        response = client.post("/chat/stream", json={"question": "问题"})
        assert response.status_code == expected_status
        assert "private-upstream-error" not in response.text
        assert "test-key" not in response.text
        if ending is not None:
            blocks = response.text.strip().split("\n\n")
            names = [block.splitlines()[0].removeprefix("event: ") for block in blocks]
            assert names[-1] == ending
            assert names.count("done") == (1 if ending == "done" else 0)
            if ending == "done":
                tokens = [json.loads(block.splitlines()[1].removeprefix("data: "))["token"]
                          for block in blocks if block.startswith("event: token\n")]
                assert "".join(tokens) == "".join(deltas)
        else:
            assert "event: done" not in response.text
        assert body.closed


@pytest.mark.parametrize("status, text", [("completed", ""), ("completed", " \t\n"), ("incomplete", "部分回答")])
def test_nonstream_incomplete_or_blank_answer_is_rejected(monkeypatch, status, text):
    payload = response_payload()
    payload["status"] = status
    payload["output"][0]["content"][0]["text"] = text
    llm = make_llm(monkeypatch, lambda request: httpx2.Response(200, json=payload))
    try:
        with pytest.raises(RuntimeError):
            llm.generate("fixture prompt")
    finally:
        llm.close()


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_nonstream_blank_answer_returns_502_with_safe_logs(monkeypatch, caplog, text):
    import logging
    from fastapi.testclient import TestClient
    from app.pipeline import RAGPipeline
    from app.ingestion.document import Document
    from app.retrieval.retriever import SearchResult
    from main import create_app

    class EvidenceRetriever:
        def search(self, question):
            return [SearchResult(Document("private-fixture-document", {"source": "test.txt", "chunk_id": 0}), 0.9, 1)]

    payload = response_payload()
    payload["output"][0]["content"][0]["text"] = text
    llm = make_llm(monkeypatch, lambda request: httpx2.Response(200, json=payload))
    logger = logging.getLogger("rag")
    logger.addHandler(caplog.handler)
    try:
        with TestClient(create_app(RAGPipeline(EvidenceRetriever(), llm))) as client:
            response = client.post("/chat", json={"question": "private-fixture-question"})
        assert response.status_code == 502
        assert any(record.name == "rag" and json.loads(record.getMessage()).get("exception_type") == "RuntimeError"
                   for record in caplog.records)
        for secret in ["test-key", "private-fixture-question", "private-fixture-document", "System:", "Traceback"]:
            assert secret not in response.text and secret not in caplog.text
    finally:
        logger.removeHandler(caplog.handler)


def test_sdk_read_timeout_is_not_returned_as_empty_answer(monkeypatch):
    from openai import APITimeoutError

    attempts = []

    def timeout(request):
        attempts.append(1)
        raise httpx2.ReadTimeout("simulated private timeout", request=request)

    monkeypatch.setattr("openai._base_client.time.sleep", lambda seconds: None)
    llm = make_llm(monkeypatch, timeout)
    try:
        with pytest.raises(APITimeoutError):
            llm.generate("fixture prompt")
        assert len(attempts) == 3  # 当前 SDK 的两次有限重试；没有真实网络请求。
    finally:
        llm.close()


@pytest.mark.parametrize("status_code, exception_type", [(401, "AuthenticationError"), (500, "InternalServerError")])
def test_sdk_error_response_is_sanitized_by_api_and_logs(monkeypatch, caplog, status_code, exception_type):
    import logging
    from fastapi.testclient import TestClient
    from app.pipeline import RAGPipeline
    from app.ingestion.document import Document
    from app.retrieval.retriever import SearchResult
    from main import create_app

    class EvidenceRetriever:
        def search(self, question):
            return [SearchResult(Document("private-sdk-evidence", {"source": "test.txt", "chunk_id": 0}), 0.9, 1)]

    monkeypatch.setattr("openai._base_client.time.sleep", lambda seconds: None)
    llm = make_llm(monkeypatch, lambda request: httpx2.Response(
        status_code, json={"error": {"message": "private-sdk-response-test-key", "type": "fixture_error"}},
    ))
    logger = logging.getLogger("rag")
    logger.addHandler(caplog.handler)
    try:
        with TestClient(create_app(RAGPipeline(EvidenceRetriever(), llm))) as client:
            response = client.post("/chat", json={"question": "private-sdk-question"})
        assert response.status_code == 502
        assert any(record.name == "rag" and json.loads(record.getMessage()).get("exception_type") == exception_type
                   for record in caplog.records)
        for secret in ["test-key", "private-sdk-response-test-key", "private-sdk-question", "private-sdk-evidence", "System:", "Traceback"]:
            assert secret not in response.text and secret not in caplog.text
    finally:
        logger.removeHandler(caplog.handler)
