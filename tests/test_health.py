"""健康接口只观察生命周期，不执行任何模型推理。"""

import asyncio
from threading import Event, Timer
from time import monotonic, sleep

import anyio
import httpx
from fastapi.testclient import TestClient
import pytest

from app.generation.llm import BaseLLM
from app.pipeline import RAGPipeline
from main import create_app


def empty_pipeline():
    class NoInferenceLLM(BaseLLM):
        closed = 0

        def generate(self, prompt):
            raise AssertionError("health must not generate")

        def stream_generate(self, prompt):
            raise AssertionError("health must not stream")

        def close(self):
            self.closed += 1

    class EmptyRetriever:
        def search(self, question):
            return []

    class NoInferenceReranker:
        def rerank(self, *args, **kwargs):
            raise AssertionError("health must not rerank")

    return RAGPipeline(EmptyRetriever(), NoInferenceLLM(), reranker=NoInferenceReranker())


def wait_for_status(client, status):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        response = client.get("/ready")
        if response.json().get("status") == status:
            return response
        sleep(.01)
    raise AssertionError(f"ready did not become {status}")


def test_health_and_readiness_before_lifespan_and_openapi():
    app = create_app(empty_pipeline())
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["status"] == "initializing"
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/health"] and "get" in paths["/ready"]


def test_initialized_empty_knowledge_base_is_ready_and_closes_once():
    pipeline = empty_pipeline()
    app = create_app(pipeline)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json() == {"status": "ready"}
        assert pipeline.llm.closed == 0
        assert client.post("/chat", json={"question": "未知问题"}).json()["status"] == "no_evidence"
    assert pipeline.llm.closed == 1
    assert app.state.initialization_status == "stopped"


def test_background_initialization_keeps_health_responsive_and_blocks_business():
    entered, release = Event(), Event()
    pipeline = empty_pipeline()

    def initialize():
        entered.set()
        assert release.wait(timeout=5)
        return pipeline

    app = create_app(pipeline_factory=initialize)
    with TestClient(app) as client:
        try:
            assert entered.wait(timeout=2)
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 503
            for path in ("/chat", "/chat/stream"):
                assert client.post(path, json={"question": "问题"}).status_code == 503
            assert client.post("/documents/upload", files={"file": ("a.txt", b"test")}).status_code == 503
        finally:
            release.set()
        assert wait_for_status(client, "ready").status_code == 200
        assert app.state.pipeline is pipeline
    assert pipeline.llm.closed == 1


def test_initialization_failure_is_unready_and_does_not_expose_exception_text():
    def unavailable():
        raise RuntimeError("fake-secret-key-and-private-path")

    with TestClient(create_app(pipeline_factory=unavailable)) as client:
        response = wait_for_status(client, "failed")
        assert response.status_code == 503
        assert "fake-secret" not in response.text
        assert client.get("/health").status_code == 200
        assert client.post("/chat", json={"question": "问题"}).status_code == 503


def test_shutdown_waits_for_initialization_and_releases_resources():
    entered, release = Event(), Event()
    pipeline = empty_pipeline()

    def initialize():
        entered.set()
        assert release.wait(timeout=5)
        return pipeline

    app = create_app(pipeline_factory=initialize)
    with TestClient(app):
        assert entered.wait(timeout=2)
        timer = Timer(.05, release.set)
        timer.start()
    timer.join(timeout=1)
    assert pipeline.llm.closed == 1 and app.state.initialization_status == "stopped"


@pytest.mark.parametrize("both", [False, True])
def test_application_requires_exactly_one_pipeline_source(both):
    with pytest.raises(ValueError):
        if both:
            create_app(empty_pipeline(), pipeline_factory=empty_pipeline)
        else:
            create_app()


def test_unready_upload_is_rejected_before_multipart_body_is_read():
    app = create_app(pipeline_factory=empty_pipeline)
    messages = []
    consumed = 0

    async def receive():
        nonlocal consumed
        consumed += 1
        return {"type": "http.request", "body": b"untrusted-body", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/documents/upload", "raw_path": b"/documents/upload",
             "query_string": b"", "root_path": "", "headers": [(b"content-type", b"multipart/form-data; boundary=test")],
             "client": ("127.0.0.1", 12345), "server": ("test", 80)}
    asyncio.run(app(scope, receive, send))
    response = next(message for message in messages if message["type"] == "http.response.start")
    assert response["status"] == 503
    assert dict(response["headers"])[b"x-request-id"]
    assert consumed == 0


def test_probes_do_not_wait_for_saturated_request_thread_pool():
    app = create_app(empty_pipeline())
    entered, release = Event(), Event()

    def blocking_request():
        entered.set()
        assert release.wait(timeout=5)

    async def check():
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = 1
        occupied = asyncio.create_task(anyio.to_thread.run_sync(blocking_request))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    responses = await asyncio.wait_for(asyncio.gather(client.get("/health"), client.get("/ready")), timeout=1)
                    assert [response.status_code for response in responses] == [200, 200]
        finally:
            release.set()
            await occupied
            limiter.total_tokens = previous

    asyncio.run(check())
