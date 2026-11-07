import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
from threading import Barrier

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app.generation.llm import BaseLLM
from app.ingestion.document import Document
from app.pipeline import RAGPipeline
from app.retrieval.retriever import SearchResult
from main import create_app


PRIVATE_KEY = "fake-api-key-DO-NOT-LOG"
PRIVATE_QUESTION = "private-user-question-DO-NOT-LOG"
PRIVATE_OUTPUT = "private-model-answer-DO-NOT-LOG"
PRIVATE_DOCUMENT = "private-document-body-DO-NOT-LOG"


@pytest.fixture
def logs(caplog):
    logger = logging.getLogger("rag")
    logger.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO, logger="rag"):
        yield caplog
    logger.removeHandler(caplog.handler)


def rows(logs):
    return [json.loads(record.getMessage()) for record in logs.records if record.name == "rag"]


def pipeline_for(mode="normal", barrier=None):
    class Retriever:
        def search(self, question):
            if barrier:
                barrier.wait(timeout=10)
            if mode == "no_evidence" or question.endswith("empty"):
                return []
            return [SearchResult(Document(PRIVATE_DOCUMENT, {"source": "private-source.txt", "chunk_id": 0}), 0.9, 1)]

    class Reranker:
        accepts_results = True

        def rerank(self, question, candidates, top_k):
            if mode == "reranker_load":
                raise OSError(PRIVATE_KEY)
            if mode == "reranker_score":
                raise RuntimeError(PRIVATE_KEY)
            return candidates[:top_k]

    class LLM(BaseLLM):
        def generate(self, prompt):
            assert PRIVATE_DOCUMENT in prompt and PRIVATE_QUESTION in prompt
            if mode == "timeout":
                raise APITimeoutError(request=httpx2.Request("POST", "https://example.test/private"))
            return PRIVATE_OUTPUT

        def stream_generate(self, prompt):
            if mode == "empty_answer":
                return
            if mode == "white_answer":
                yield " \t\n"
                return
            if mode == "after_start":
                yield "partial-private-output"
                raise RuntimeError(PRIVATE_KEY)
            yield self.generate(prompt)

    return RAGPipeline(Retriever(), LLM(), reranker=Reranker(), top_k=1)


@pytest.mark.parametrize("path", ["/chat", "/chat/stream", "/missing"])
def test_request_id_in_response_and_json_stage_logs(logs, path):
    with TestClient(create_app(pipeline_for())) as client:
        response = client.get(path) if path == "/missing" else client.post(path, json={"question": PRIVATE_QUESTION})
    request_id = response.headers.get("X-Request-ID", "")
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    events = rows(logs)
    assert events and all(event["request_id"] == request_id for event in events)
    terminal = [event for event in events if event["stage"] == "http" and event["event"] != "stage_started"]
    assert len(terminal) == 1
    assert terminal[0]["http_status"] == response.status_code
    assert terminal[0]["duration_ms"] >= 0
    if path != "/missing":
        for stage in ["pipeline", "retrieval", "reranker", "llm"]:
            assert any(event["stage"] == stage and event["event"] == "stage_completed" for event in events)


@pytest.mark.parametrize("supplied", ["caller-supplied-id", "bad\r\nINJECTED", "x" * 1000])
def test_external_request_ids_are_not_trusted_or_logged(logs, supplied):
    with TestClient(create_app(pipeline_for())) as client:
        response = client.post("/chat", headers={"X-Request-ID": supplied}, json={"question": PRIVATE_QUESTION})
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers.get("X-Request-ID", ""))
    assert response.headers["X-Request-ID"] != supplied
    assert supplied not in logs.text


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_concurrent_requests_keep_their_own_ids_and_statuses(logs, path):
    with TestClient(create_app(pipeline_for(barrier=Barrier(2)))) as client:
        def request(suffix):
            response = client.post(path, json={"question": PRIVATE_QUESTION + suffix})
            return response.headers.get("X-Request-ID", ""), suffix

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(request, ["normal", "empty"]))
    ids = [request_id for request_id, _ in responses]
    assert len(set(ids)) == 2 and all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    events = rows(logs)
    for request_id, suffix in responses:
        own = [event for event in events if event["request_id"] == request_id]
        terminal = [event for event in own if event["stage"] == "http" and event["event"] != "stage_started"]
        assert len(terminal) == 1 and terminal[0]["http_status"] == 200
        assert terminal[0]["status"] == ("no_evidence" if suffix == "empty" else "ok")
        assert any(event["stage"] == "llm" for event in own) is (suffix == "normal")


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_no_evidence_has_status_without_llm_success(logs, path):
    with TestClient(create_app(pipeline_for("no_evidence"))) as client:
        response = client.post(path, json={"question": PRIVATE_QUESTION})
    assert response.status_code == 200
    events = rows(logs)
    assert not any(event["stage"] == "llm" for event in events)
    assert any(event["stage"] == "pipeline" and event["status"] == "no_evidence" for event in events)
    assert any(event["stage"] == "http" and event["status"] == "no_evidence" for event in events)


@pytest.mark.parametrize("mode, failed_stage, exception_type", [
    ("timeout", "llm", "APITimeoutError"),
    ("reranker_load", "reranker", "OSError"),
    ("reranker_score", "reranker", "RuntimeError"),
])
@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_failure_records_safe_exception_and_preserves_502(logs, mode, failed_stage, exception_type, path):
    with TestClient(create_app(pipeline_for(mode))) as client:
        response = client.post(path, json={"question": PRIVATE_QUESTION})
    assert response.status_code == 502
    request_id = response.headers.get("X-Request-ID", "")
    failures = [event for event in rows(logs) if event["event"] == "stage_failed"]
    assert any(event["stage"] == failed_stage and event["exception_type"] == exception_type for event in failures)
    assert failures and all(event["request_id"] == request_id for event in failures)
    if failed_stage == "reranker":
        assert not any(event["stage"] == "llm" for event in rows(logs))
    assert PRIVATE_KEY not in response.text and "Traceback" not in response.text


@pytest.mark.parametrize("mode, expected_status", [("after_start", 200), ("empty_answer", 502), ("white_answer", 502)])
def test_sse_failure_logs_include_logical_failure_even_with_http_200(logs, mode, expected_status):
    with TestClient(create_app(pipeline_for(mode))) as client:
        response = client.post("/chat/stream", json={"question": PRIVATE_QUESTION})
    assert response.status_code == expected_status
    assert "event: done" not in response.text
    if mode == "after_start":
        assert "event: token" in response.text and "event: error" in response.text
    events = rows(logs)
    assert any(event["event"] == "stage_failed" and event["exception_type"] == "RuntimeError" for event in events)
    terminal = [event for event in events if event["stage"] == "http" and event["event"] != "stage_started"]
    assert len(terminal) == 1 and terminal[0]["status"] == "failed"
    assert terminal[0]["http_status"] == expected_status


@pytest.mark.parametrize("mode", ["normal", "reranker_score", "after_start"])
def test_logs_do_not_include_private_content_or_exception_messages(logs, mode):
    with TestClient(create_app(pipeline_for(mode))) as client:
        client.post("/chat/stream", headers={"Authorization": "Bearer " + PRIVATE_KEY}, json={"question": PRIVATE_QUESTION})
    assert rows(logs)
    for secret in [PRIVATE_KEY, PRIVATE_QUESTION, PRIVATE_OUTPUT, PRIVATE_DOCUMENT, "partial-private-output", "System:"]:
        assert secret not in logs.text
    assert all(record.exc_info is None and record.stack_info is None for record in logs.records if record.name == "rag")


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize("mode", ["normal", "no_evidence"])
def test_logging_failure_does_not_change_response(monkeypatch, path, mode):
    def fail(*args, **kwargs):
        raise OSError("logging sink unavailable " + PRIVATE_KEY)

    monkeypatch.setattr(logging.getLogger("rag"), "log", fail)
    with TestClient(create_app(pipeline_for(mode))) as client:
        response = client.post(path, json={"question": PRIVATE_QUESTION})
    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers.get("X-Request-ID", ""))
    assert PRIVATE_KEY not in response.text
    if mode == "normal":
        assert PRIVATE_OUTPUT in response.text
    else:
        assert "no_evidence" in response.text


def test_stream_close_records_cancellation_and_releases_upstream(logs):
    class TrackingLLM(BaseLLM):
        closed = False

        def stream_generate(self, prompt):
            try:
                yield PRIVATE_OUTPUT
                yield "more"
            finally:
                self.closed = True

    pipeline = pipeline_for()
    pipeline.llm = TrackingLLM()
    stream = pipeline.stream_ask(PRIVATE_QUESTION)
    assert next(stream) == PRIVATE_OUTPUT
    stream.close()
    assert pipeline.llm.closed
    assert any(event["stage"] == "llm" and event["status"] == "cancelled" for event in rows(logs))


def test_http_duration_includes_sse_send_completion(logs):
    app = create_app(pipeline_for())
    body = json.dumps({"question": PRIVATE_QUESTION}).encode()
    messages = []

    async def invoke():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                assert not any(event["stage"] == "http" and event["event"] != "stage_started" for event in rows(logs))

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/chat/stream", "raw_path": b"/chat/stream", "query_string": b"",
                 "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
        await app(scope, receive, send)

    asyncio.run(invoke())
    terminal = [event for event in rows(logs) if event["stage"] == "http" and event["event"] != "stage_started"]
    assert len(terminal) == 1 and terminal[0]["duration_ms"] >= 0


def test_invalid_request_keeps_id_without_logging_input(logs):
    with TestClient(create_app(pipeline_for())) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422
    request_id = response.headers.get("X-Request-ID", "")
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert not any(event["stage"] == "pipeline" for event in rows(logs))
    assert any(event["stage"] == "http" and event["status"] == "rejected" for event in rows(logs))


def test_unhandled_500_response_still_returns_id_and_safe_body(logs):
    app = create_app(pipeline_for())

    @app.get("/unexpected")
    def fail():
        raise RuntimeError(PRIVATE_KEY)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/unexpected")
    assert response.status_code == 500
    request_id = response.headers.get("X-Request-ID", "")
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert PRIVATE_KEY not in response.text and PRIVATE_KEY not in logs.text
    assert any(event["stage"] == "http" and event["request_id"] == request_id and event.get("exception_type") == "RuntimeError"
               for event in rows(logs))


def test_formatter_failure_does_not_print_sink_error_or_change_business(monkeypatch, capsys):
    app = create_app(pipeline_for())
    handler = next(handler for handler in logging.getLogger("rag").handlers if getattr(handler, "_rag_json", False))

    class BrokenFormatter(logging.Formatter):
        def format(self, record):
            raise OSError(PRIVATE_KEY)

    monkeypatch.setattr(handler, "formatter", BrokenFormatter())
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": PRIVATE_QUESTION})
    assert response.status_code == 200 and response.json()["answer"] == PRIVATE_OUTPUT
    assert PRIVATE_KEY not in capsys.readouterr().err


def test_request_context_resets_after_unhandled_error(logs):
    from app.observability import request_trace

    app = create_app(pipeline_for())

    @app.get("/unexpected")
    def fail():
        raise RuntimeError(PRIVATE_KEY)

    async def invoke():
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
                 "method": "GET", "scheme": "http", "path": "/unexpected", "raw_path": b"/unexpected", "query_string": b"",
                 "headers": [], "server": ("test", 80), "client": ("test", 123)}
        assert request_trace.get() is None
        with pytest.raises(RuntimeError):
            await app(scope, receive, send)
        assert request_trace.get() is None

    asyncio.run(invoke())


@pytest.mark.parametrize("mode, exception_type", [("timeout", APITimeoutError), ("reranker_score", RuntimeError)])
@pytest.mark.parametrize("streaming", [False, True])
def test_logging_failure_preserves_original_business_exception(monkeypatch, mode, exception_type, streaming):
    def fail(*args, **kwargs):
        raise OSError("logging sink failed")

    monkeypatch.setattr(logging.getLogger("rag"), "log", fail)
    pipeline = pipeline_for(mode)
    with pytest.raises(exception_type):
        if streaming:
            list(pipeline.stream_ask(PRIVATE_QUESTION))
        else:
            pipeline.ask(PRIVATE_QUESTION)


def test_exception_type_cannot_inject_external_text_into_logs(logs):
    injected_type = type("Injected\n" + PRIVATE_KEY, (RuntimeError,), {})
    pipeline = pipeline_for()

    def fail(prompt):
        raise injected_type(PRIVATE_KEY)

    pipeline.llm.generate = fail
    with TestClient(create_app(pipeline)) as client:
        response = client.post("/chat", json={"question": PRIVATE_QUESTION})
    assert response.status_code == 502
    assert PRIVATE_KEY not in logs.text
    assert any(event.get("exception_type") == "Exception" for event in rows(logs))
