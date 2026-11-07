from copy import deepcopy
import json

import httpx2
from openai import OpenAI
import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.qa_dataset import load_qa_dataset
from app.generation.llm import BaseLLM, OpenAILLM
from app.retrieval.retriever import SearchResult
from config import Settings
from scripts.run_v3_comprehensive import (
    InFlightError, LLMMonitor, ProgressStore, run_cell, should_stop,
    summarize_units, validate_provider, aggregate_qa, publish_text, publish_review_bundles,
)


def test_progress_does_not_repeat_completed_or_ambiguous_calls(tmp_path):
    store = ProgressStore(tmp_path, "run-a")
    cell, reused = store.begin("vector__qa_001", {"qa_id": "qa_001", "method": "vector"})
    assert reused is False
    with pytest.raises(InFlightError):
        store.begin("vector__qa_001", {})
    cell.update(state="finished", status="failed", exception_type="APITimeoutError")
    store.update(cell)
    saved, reused = store.begin("vector__qa_001", {})
    assert reused is True and saved["status"] == "failed"
    with pytest.raises(ValueError):
        ProgressStore(tmp_path, "run-b").begin("vector__qa_001", {})


def test_progress_checksum_detects_modified_completion(tmp_path):
    store = ProgressStore(tmp_path, "run-a")
    cell, _ = store.begin("vector__qa_001", {})
    cell.update(state="finished", status="success")
    store.update(cell)
    path = tmp_path / "vector__qa_001.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["status"] = "failed"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        store.begin("vector__qa_001", {})


def test_finished_units_are_immutable_even_with_a_valid_digest(tmp_path):
    store = ProgressStore(tmp_path, "run-a")
    cell, _ = store.begin("vector__qa_001", {})
    cell.update(state="finished", status="success")
    store.update(cell)
    before = (tmp_path / "vector__qa_001.json").read_bytes()
    cell["status"] = "failed"
    with pytest.raises(ValueError, match="immutable"):
        store.update(cell)
    assert (tmp_path / "vector__qa_001.json").read_bytes() == before
    assert not list(tmp_path.glob("*.lock"))


def test_stale_checkpoint_writer_cannot_erase_a_newer_attempt(tmp_path):
    store = ProgressStore(tmp_path, "run-a")
    cell, _ = store.begin("vector__qa_001", {})
    stale = deepcopy(cell)
    cell["http_attempts"].append({"attempt": 1, "http_status": None})
    store.update(cell)
    with pytest.raises(ValueError, match="Stale"):
        store.update(stale)
    assert len(store.load("vector__qa_001")["http_attempts"]) == 1


def test_provider_guard_never_falls_back_or_exposes_a_key():
    assert validate_provider(Settings(model_name="deepseek-flash", api_key="secret", base_url="https://api.deepseek.com"))["model"] == "deepseek-flash"
    for settings in [Settings(), Settings("gpt-test", "secret", "https://api.openai.com"), Settings("deepseek-flash", "secret", "https://user:secret@api.deepseek.com")]:
        with pytest.raises(ValueError) as error:
            validate_provider(settings)
        assert "secret" not in str(error.value)


@pytest.fixture
def data():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    corpus = load_dataset(root / "data/evaluation/retrieval_eval_v3.json")
    qa = load_qa_dataset(root / "data/evaluation/qa_eval_v3.json", corpus)
    return corpus, qa


class CountingLLM(BaseLLM):
    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return "只依据实际资料的测试输出"


def test_real_pipeline_collection_has_no_label_leak_and_resume_is_free(tmp_path, data):
    corpus, qa = data
    sample = deepcopy(qa.samples[0])
    sample.update(reference_answer="REFERENCE_DO_NOT_SEND", required_facts=["FACT_DO_NOT_SEND"])
    document = corpus.documents[0]

    class Retriever:
        def search(self, query, top_k):
            assert top_k == 10
            return [SearchResult(document, .8, 1)]

    llm = CountingLLM()
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    unit = run_cell("vector", sample, corpus, qa, Retriever(), None, llm, monitor, store)
    assert unit["status"] == "success"
    assert document.text in llm.calls[0]
    assert "REFERENCE_DO_NOT_SEND" not in llm.calls[0] and "FACT_DO_NOT_SEND" not in llm.calls[0]
    assert unit["record"]["context_document_ids"] == [document.metadata["eval_id"]]
    assert unit["record"]["review"]["faithfulness"] is None
    assert run_cell("vector", sample, corpus, qa, Retriever(), None, llm, monitor, store) == unit
    assert len(llm.calls) == 1


def test_empty_candidates_keep_no_evidence_without_model_call(tmp_path, data):
    corpus, qa = data

    class Empty:
        def search(self, query, top_k):
            return []

    llm = CountingLLM()
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    unit = run_cell("bm25", qa.samples[0], corpus, qa, Empty(), None, llm, monitor, store)
    assert unit["status"] == "no_evidence" and llm.calls == []
    assert unit["record"]["pipeline_status"] == "no_evidence"
    assert unit["record"]["prompt"] is None
    assert unit["http_attempts"] == []


def test_interrupt_during_generation_remains_ambiguous_and_cannot_be_replayed(tmp_path, data):
    corpus, qa = data

    class Retriever:
        def search(self, query, top_k):
            return [SearchResult(corpus.documents[0], .8, 1)]

    class InterruptingLLM(BaseLLM):
        def generate(self, prompt):
            raise KeyboardInterrupt()

    llm = InterruptingLLM()
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    with pytest.raises(KeyboardInterrupt):
        run_cell("vector", qa.samples[0], corpus, qa, Retriever(), None, llm, monitor, store)
    cell = store.load("vector__qa_001")
    assert cell["state"] == "in_flight" and "status" not in cell
    with pytest.raises(InFlightError):
        store.begin("vector__qa_001", {})


def test_reranker_only_gets_actual_top10_candidates(tmp_path, data):
    corpus, qa = data
    candidates = [SearchResult(d, .5, i) for i, d in enumerate(corpus.documents[:10], 1)]

    class Retriever:
        def search(self, query, top_k):
            return candidates

    class Reranker:
        accepts_results = True

        def rerank(self, query, results, top_k):
            assert results == candidates and top_k == 3
            return list(reversed(results))[:3]

    llm = CountingLLM()
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    unit = run_cell("hybrid_reranker", qa.samples[0], corpus, qa, Retriever(), Reranker(), llm, monitor, store)
    assert len(unit["retrieval_candidates"]) == 10
    assert unit["record"]["context_document_ids"] == [d.metadata["eval_id"] for d in corpus.documents[7:10]][::-1]


def response_payload():
    return {"id": "resp-test", "object": "response", "created_at": 0, "status": "completed", "model": "deepseek-flash",
            "output": [{"id": "m-test", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "实际 SDK 测试答案", "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}}


@pytest.mark.parametrize("status", [200, 401])
def test_sdk_monitor_records_usage_and_safe_failures(tmp_path, monkeypatch, status):
    import app.generation.llm as module

    def handler(request):
        return httpx2.Response(status, json=response_payload() if status == 200 else {"error": {"message": "secret-key-private", "type": "authentication_error"}})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=client))
    llm = OpenAILLM(settings=Settings("deepseek-flash", "fake-secret", "https://example.test"))
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    cell, _ = store.begin("vector__qa_001", {})
    monitor.activate(cell)
    try:
        if status == 200:
            assert llm.generate("test prompt") == "实际 SDK 测试答案"
            assert cell["usage"]["total_tokens"] == 25
        else:
            with pytest.raises(Exception):
                llm.generate("test prompt")
        assert len(cell["http_attempts"]) == 1
        assert cell["http_attempts"][0]["http_status"] == status
        serialized = json.dumps(store.load("vector__qa_001"), ensure_ascii=False)
        assert "fake-secret" not in serialized and "secret-key-private" not in serialized
    finally:
        llm.close()


def test_sdk_retries_are_counted_once_and_usage_is_not_duplicated(tmp_path, monkeypatch):
    import app.generation.llm as module
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx2.Response(500, json={"error": {"message": "private"}}) if len(attempts) < 3 else httpx2.Response(200, json=response_payload())

    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))))
    llm = OpenAILLM(settings=Settings("deepseek-flash", "fake", "https://example.test"))
    monkeypatch.setattr(llm.client, "_sleep_for_retry", lambda **kwargs: None)
    store = ProgressStore(tmp_path, "run-a")
    monitor = LLMMonitor(llm, store)
    cell, _ = store.begin("vector__qa_001", {})
    monitor.activate(cell)
    try:
        llm.generate("prompt")
        assert [a["http_status"] for a in cell["http_attempts"]] == [500, 500, 200]
        assert cell["usage"]["total_tokens"] == 25
    finally:
        llm.close()


def test_stop_and_summary_do_not_invent_quality_or_token_counts():
    assert should_stop([{"status": "failed", "http_status": 401}])
    assert should_stop([{"status": "failed", "http_status": 500}] * 3)
    assert not should_stop([{"status": "failed", "http_status": 500}, {"status": "success"}])
    summary = summarize_units([{"method": "vector", "status": "failed", "http_attempts": [{"http_status": 500}] * 3,
                                "usage": None, "timings_ms": {"total": 10}}])
    assert summary["vector"]["http_requests"] == 3 and summary["vector"]["retries"] == 2
    assert summary["vector"]["usage_available_units"] == 0
    assert summary["vector"]["human_quality_metrics"] is None


def test_text_publication_refuses_overwrite_and_cleans_failed_publish(tmp_path, monkeypatch):
    import scripts.run_v3_comprehensive as runner
    path = tmp_path / "report.txt"
    publish_text("original", path)
    with pytest.raises(FileExistsError):
        publish_text("replacement", path)
    assert path.read_text(encoding="utf-8") == "original"
    monkeypatch.setattr(runner.os, "link", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError):
        publish_text("new content", tmp_path / "new.txt")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.txt"]


def test_method_review_files_reuse_day8_validation_without_quality_scores(tmp_path, data):
    from app.evaluation.qa_review import load_qa_results, summarize_reviews
    corpus, qa = data

    class Retriever:
        def search(self, query, top_k):
            return [SearchResult(corpus.documents[0], .8, 1)]

    llm = CountingLLM()
    store = ProgressStore(tmp_path / "units", "run-a")
    monitor = LLMMonitor(llm, store)
    unit = run_cell("vector", qa.samples[0], corpus, qa, Retriever(), None, llm, monitor, store)
    report = aggregate_qa([unit], {"experiment_id": "run-a", "config": {}, "initialization": {}})
    publish_review_bundles(report, tmp_path)
    bundle = load_qa_results(tmp_path / "v3_qa_review/vector.json")
    summary = summarize_reviews(bundle)
    assert summary["reviewed_count"] == 0 and summary["human_review_metrics"] is None
    assert summary["real_llm_quality_metrics"] is None
    assert len(llm.calls) == 1


@pytest.mark.parametrize("http_status, expected_calls, exit_code", [(200, 200, 0), (401, 1, 2)])
def test_offline_whole_batch_and_completed_resume_do_not_repeat_http_requests(tmp_path, data, monkeypatch, http_status, expected_calls, exit_code):
    import app.generation.llm as llm_module
    import scripts.run_v3_comprehensive as runner
    from pathlib import Path
    import sys

    corpus, qa = data
    source_root = Path(__file__).resolve().parents[1]
    (tmp_path / "data/evaluation").mkdir(parents=True)
    (tmp_path / "app/generation").mkdir(parents=True)
    for name in ("retrieval_eval_v3.json", "qa_eval_v3.json"):
        (tmp_path / "data/evaluation" / name).write_bytes((source_root / "data/evaluation" / name).read_bytes())
    (tmp_path / "app/generation/prompt.py").write_bytes((source_root / "app/generation/prompt.py").read_bytes())
    calls, builds = [], []

    class Retriever:
        def search(self, query, top_k):
            return [SearchResult(d, .9, rank) for rank, d in enumerate(corpus.documents[:top_k], 1)]

    class Reranker:
        accepts_results = True

        def rerank(self, query, results, top_k):
            return results[:top_k]

    def models(actual_corpus):
        builds.append(1)
        class ActualRetriever:
            def search(self, query, top_k):
                return [SearchResult(d, .9, rank) for rank, d in enumerate(actual_corpus.documents[:top_k], 1)]
        return {m: ActualRetriever() for m in runner.METHODS}, Reranker(), {"total_ms": 0}

    def handler(request):
        payload = json.loads(request.content)
        assert set(payload) == {"model", "input"}
        assert payload["model"] == "deepseek-flash"
        calls.append(1)
        return httpx2.Response(http_status, json=response_payload() if http_status == 200 else {"error": {"message": "private-auth-error"}})

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_models", models)
    monkeypatch.setattr(runner.Settings, "from_env", classmethod(lambda cls: Settings("deepseek-flash", "fake-secret", "https://api.deepseek.com")))
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))))
    monkeypatch.setattr(sys, "argv", ["run_v3_comprehensive.py"])
    assert runner.main() == exit_code
    assert len(calls) == expected_calls
    path = tmp_path / "results/v3_qa_comparison.json"
    saved = path.read_bytes()
    result = json.loads(saved)
    assert result["finished_units"] == expected_calls
    assert result["human_review"]["quality_metrics"] is None
    assert b"fake-secret" not in saved and b"private-auth-error" not in saved
    monkeypatch.setattr(sys, "argv", ["run_v3_comprehensive.py", "--resume"])
    assert runner.main() == exit_code
    assert len(calls) == expected_calls and len(builds) == 1
    assert path.read_bytes() == saved
