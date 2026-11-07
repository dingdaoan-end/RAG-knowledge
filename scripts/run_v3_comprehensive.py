"""V3 独立实验入口；复用生产 Pipeline/Provider 和既有评估，不修改默认入口。"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from statistics import fmean, median
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_dataset
from app.evaluation.evaluator import METHODS, SCORE_FIELDS, evaluate_methods, format_report, save_report
from app.evaluation.qa_dataset import QADataset, load_qa_dataset
from app.evaluation.qa_evaluator import _document_info, collect_qa
from app.evaluation.qa_review import capture_digest, validate_qa_results
from app.generation.llm import OpenAILLM
from app.pipeline import RAGPipeline
from config import Settings


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


class InFlightError(RuntimeError):
    """付费调用可能已发生，不能自动重放。"""


class ProgressStore:
    def __init__(self, directory, experiment_id):
        self.directory = Path(directory)
        self.experiment_id = experiment_id

    def path(self, unit_id):
        if not unit_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in unit_id):
            raise ValueError("Invalid unit identifier")
        return self.directory / (unit_id + ".json")

    def load(self, unit_id):
        value = json.loads(self.path(unit_id).read_text(encoding="utf-8"))
        if value["experiment_id"] != self.experiment_id or value["unit_id"] != unit_id:
            raise ValueError("Experiment identity mismatch")
        if value.get("unit_digest") != digest({k: v for k, v in value.items() if k != "unit_digest"}):
            raise ValueError("Progress integrity mismatch")
        return value

    def begin(self, unit_id, fields):
        if self.path(unit_id).exists():
            value = self.load(unit_id)
            if value["state"] != "finished":
                raise InFlightError("Ambiguous interrupted unit; inspect saved progress before another paid call")
            return value, True
        value = {**fields, "experiment_id": self.experiment_id, "unit_id": unit_id,
                 "state": "in_flight", "started_at": now(), "http_attempts": [], "usage": None,
                 "timings_ms": {}, "llm_called": False}
        value["unit_digest"] = digest(value)
        save_report(value, self.path(unit_id))
        return value, False

    def update(self, value):
        path = self.path(value["unit_id"])
        lock_path = path.with_suffix(".lock")
        guard = lock_path.open("x", encoding="ascii")
        temporary = None
        try:
            current = self.load(value["unit_id"])
            if current["state"] == "finished":
                raise ValueError("Finished unit is immutable")
            if value["experiment_id"] != self.experiment_id or value.get("unit_digest") != current["unit_digest"]:
                raise ValueError("Stale checkpoint writer or experiment identity mismatch")
            next_digest = digest({k: v for k, v in value.items() if k != "unit_digest"})
            serialized = {**value, "unit_digest": next_digest}
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                    prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(serialized, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)  # 只更新本实验已声明的 checkpoint，非历史报告。
            value["unit_digest"] = next_digest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            guard.close()
            lock_path.unlink(missing_ok=True)


def validate_provider(settings):
    url = urlsplit(settings.base_url or "")
    if (not settings.api_key or not settings.model_name.startswith("deepseek-") or
            url.scheme != "https" or url.hostname != "api.deepseek.com" or
            url.username or url.password or url.query or url.fragment):
        raise ValueError("DeepSeek configuration unavailable or unexpected; no provider fallback")
    return {"provider": "DeepSeek", "model": settings.model_name, "base_url": settings.base_url}


def safe_type(error):
    name = type(error).__name__
    return name if name.isascii() and name.isidentifier() else "Exception"


class LLMMonitor:
    """仅在实验对象上观察既有 generate/create；不改请求参数或 Provider 行为。"""
    def __init__(self, llm, store):
        self.llm, self.store, self.active = llm, store, None
        self.original_generate = llm.generate
        llm.generate = self.generate
        if hasattr(llm, "client"):
            self.original_create = llm.client.responses.create
            llm.client.responses.create = self.create
            llm.client._client.event_hooks["request"].append(self.request)
            llm.client._client.event_hooks["response"].append(self.response)

    def activate(self, value):
        self.active = value

    def request(self, request):
        self.active["http_attempts"].append({"attempt": len(self.active["http_attempts"]) + 1,
                                              "started_at": now(), "http_status": None})
        self.store.update(self.active)  # 发出 HTTP 前持久化；崩溃后绝不盲目重放。

    def response(self, response):
        self.active["http_attempts"][-1]["http_status"] = response.status_code
        self.store.update(self.active)

    def create(self, *args, **kwargs):
        response = self.original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            values = {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens", "total_tokens")}
            values = {k: v for k, v in values.items() if type(v) is int}
            details = getattr(usage, "input_tokens_details", None)
            if type(getattr(details, "cached_tokens", None)) is int:
                values["cached_input_tokens"] = details.cached_tokens
            self.active["usage"] = values or None
        request_id = getattr(response, "_request_id", None)
        if isinstance(request_id, str) and request_id.isascii() and len(request_id) <= 128:
            self.active["provider_request_id"] = request_id
        self.active["returned_generation_parameters"] = {
            k: getattr(response, k) for k in ("temperature", "top_p", "max_output_tokens")
            if isinstance(getattr(response, k, None), (int, float))
        }
        self.active["provider_response_status"] = getattr(response, "status", None)
        self.active["returned_model"] = getattr(response, "model", None)
        self.store.update(self.active)
        return response

    def generate(self, prompt):
        if self.active is None:
            raise RuntimeError("Untracked model request")
        self.active.update(llm_called=True, actual_prompt=prompt)
        self.store.update(self.active)
        start = perf_counter()
        try:
            return self.original_generate(prompt)
        finally:
            self.active["timings_ms"]["llm"] = (perf_counter() - start) * 1000
            self.store.update(self.active)


def rows(results, registry):
    return [{**_document_info(result.document, registry), "rank": result.rank, "score": float(result.score),
             **{key: (float(getattr(result, key)) if getattr(result, key) is not None else None)
                for key in SCORE_FIELDS if hasattr(result, key)}} for result in results]


class TimedSearch:
    def __init__(self, delegate, registry):
        self.delegate, self.registry = delegate, registry
        self.results, self.elapsed, self.failed = [], 0.0, False
        self.calls = []

    def search(self, query, top_k=10):
        start = perf_counter()
        try:
            self.results = self.delegate.search(query, top_k=top_k)
            if len(self.results) > top_k:
                raise ValueError("Candidate budget exceeded")
            return self.results
        except Exception:
            self.failed = True
            raise
        finally:
            self.elapsed = (perf_counter() - start) * 1000
            self.calls.append({"query": query, "duration_ms": self.elapsed})


class TimedReranker:
    accepts_results = True

    def __init__(self, delegate):
        self.delegate = delegate
        self.results, self.elapsed, self.failed = [], 0.0, False
        self.calls = []
        self.model_name = getattr(delegate, "model_name", None)

    def rerank(self, query, candidates, top_k):
        start = perf_counter()
        try:
            self.results = self.delegate.rerank(query, candidates, top_k=top_k)
            if {id(r.document) for r in self.results} - {id(r.document) for r in candidates}:
                raise ValueError("Reranker introduced an unretrieved candidate")
            return self.results
        except Exception:
            self.failed = True
            raise
        finally:
            self.elapsed = (perf_counter() - start) * 1000
            self.calls.append({"query": query, "duration_ms": self.elapsed})


def run_cell(method, sample, corpus, qa, retriever, reranker, llm, monitor, store):
    unit, reused = store.begin(method + "__" + sample["id"], {
        "method": method, "qa_id": sample["id"], "query_id": sample["id"], "question": sample["question"],
        "retrieval_eval_query_ids": [q["id"] for q in corpus.queries if q["query"] == sample["question"]],
    })
    if reused:
        return unit
    registry = {d.metadata["eval_id"]: d for d in corpus.documents}
    search = TimedSearch(retriever, registry)
    ranker = TimedReranker(reranker) if reranker is not None else None
    monitor.activate(unit)
    start = perf_counter()
    try:
        subset = QADataset(qa.version, qa.sha256, qa.corpus_sha256, [sample], qa.metadata)
        bundle = collect_qa(RAGPipeline(search, llm, reranker=ranker, top_k=3), subset, corpus)
        record = bundle["records"][0]
        unit.update(status="no_evidence" if record["pipeline_status"] == "no_evidence" else "success",
                    record=record, bundle_run=bundle["run"], exception_type=None,
                    pipeline_status=record["pipeline_status"], failed_stage=None)
    except Exception as error:
        unit.update(status="failed", record=None, pipeline_status=None, exception_type=safe_type(error),
                    http_status=getattr(error, "status_code", None),
                    failed_stage="retrieval" if search.failed else "reranker" if ranker and ranker.failed else "llm" if unit["llm_called"] else "collection")
    finally:
        unit["timings_ms"].update(retrieval=search.elapsed, rerank=ranker.elapsed if ranker else None,
                                   total=(perf_counter() - start) * 1000)
        unit["retrieval_candidates"] = rows(search.results, registry)
        unit["reranked_candidates"] = rows(ranker.results, registry) if ranker else []
        unit["retry_count"] = max(0, len(unit["http_attempts"]) - 1)
        if "status" in unit:
            unit.update(state="finished", finished_at=now())
        # KeyboardInterrupt/SystemExit 可能发生在付费请求中；保持 in_flight，禁止自动重放。
        store.update(unit)
        monitor.active = None
    return unit


def should_stop(units):
    if not units:
        return False
    last = units[-1]
    if last.get("http_status") in {400, 401, 402, 403, 404, 429}:
        return True
    if last.get("exception_type") in {"ValueError", "AssertionError"}:
        return True
    return len(units) >= 3 and all(u["status"] == "failed" for u in units[-3:])


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    position = .95 * (len(values) - 1)
    lo = int(position)
    p95 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)
    return {"n": len(values), "mean": fmean(values), "median": median(values), "p95": p95,
            "min": values[0], "max": values[-1]}


def summarize_units(units):
    summary = {}
    for method in METHODS:
        selected = [u for u in units if u["method"] == method]
        usage = [u["usage"] for u in selected if u.get("usage") is not None]
        summary[method] = {
            "units": len(selected), "statuses": dict(Counter(u["status"] for u in selected)),
            "http_requests": sum(len(u.get("http_attempts", [])) for u in selected),
            "retries": sum(max(0, len(u.get("http_attempts", [])) - 1) for u in selected),
            "http_response_statuses": dict(Counter(str(a["http_status"]) for u in selected for a in u.get("http_attempts", []))),
            "usage_available_units": len(usage),
            "tokens": {k: sum(u.get(k, 0) for u in usage) if any(k in u for u in usage) else None
                       for k in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")},
            "timings_ms": {k: distribution([u.get("timings_ms", {}).get(k) for u in selected])
                           for k in ("retrieval", "rerank", "llm", "total")},
            "steady_total_ms_excluding_first_unit": distribution([u.get("timings_ms", {}).get("total") for u in selected[1:]]),
            "human_quality_metrics": None,
        }
    return summary


def build_models(corpus):
    started = perf_counter()
    from huggingface_hub import snapshot_download
    from app.embedding.embedder import TextEmbedder
    from app.retrieval.retriever import FaissRetriever
    from app.retrieval.text_retriever import TextRetriever
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever
    from app.retrieval.reranker import CrossEncoderReranker
    import torch

    names = ("BAAI/bge-small-zh-v1.5", "BAAI/bge-reranker-base")
    paths = {name: snapshot_download(name, local_files_only=True) for name in names}
    stamp = perf_counter()
    embedder = TextEmbedder(model_name=paths[names[0]], batch_size=32)
    embedding_init_ms = (perf_counter() - stamp) * 1000
    stamp = perf_counter()
    vector = TextRetriever(FaissRetriever(corpus.documents, embedder.embed_documents(corpus.documents)), embedder, top_k=10)
    bm25 = BM25Retriever(corpus.documents, top_k=10)
    hybrid = HybridRetriever(vector, bm25, alpha=.7, top_k=10)
    index_init_ms = (perf_counter() - stamp) * 1000
    stamp = perf_counter()
    reranker = CrossEncoderReranker(model_name=paths[names[1]], local_files_only=True)
    reranker._get_scorer()  # 单独计冷启动；之后的重排只对 Top-10 计时。
    initialization = {"embedding_init_ms": embedding_init_ms, "index_init_ms": index_init_ms,
                      "reranker_init_ms": (perf_counter() - stamp) * 1000,
                      "total_ms": (perf_counter() - started) * 1000,
                      "models": {name: {"local_path": path, "revision": Path(path).name} for name, path in paths.items()},
                      "torch_threads": torch.get_num_threads(), "device": str(embedder.model.device)}
    return {"vector": vector, "bm25": bm25, "hybrid": hybrid, "hybrid_reranker": hybrid}, reranker, initialization


def publish_text(text, path):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing output differs; choose a new experiment path")
        return
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".text-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # 完整内容才发布；不会覆盖已有目标。
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_retrieval(corpus, retrievers, reranker, manifest, output):
    path = output / "v3_retrieval_comparison.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("run", {}).get("experiment_id") != manifest["experiment_id"] or report["dataset"]["sha256"] != corpus.sha256:
            raise ValueError("Existing retrieval report identity mismatch")
    else:
        registry = {d.metadata["eval_id"]: d for d in corpus.documents}
        timed = {method: TimedSearch(retrievers[method], registry) for method in METHODS[:3]}
        ranker = TimedReranker(reranker)
        print(json.dumps({"event": "retrieval_started", "queries": len(corpus.queries)}), flush=True)
        report = evaluate_methods(corpus, timed["vector"], timed["bm25"], timed["hybrid"], ranker, ks=(3, 5, 10), candidate_k=10)
        for index, sample in enumerate(report["queries"]):
            for method in METHODS:
                base = "hybrid" if method == "hybrid_reranker" else method
                sample["methods"][method]["timings_ms"] = {
                    "retrieval": timed[base].calls[index]["duration_ms"],
                    "rerank": ranker.calls[index]["duration_ms"] if method == "hybrid_reranker" else None,
                }
        report["run"] = {"experiment_id": manifest["experiment_id"], "created_at": now(), "config": manifest["config"],
                         "initialization": manifest["initialization"]}
        save_report(report, path)
    text = format_report(report) + "\n"
    publish_text(text, output / "v3_retrieval_comparison.txt")
    print(json.dumps({"event": "retrieval_completed", "queries": len(report["queries"]), "summary": report["summary"]}), flush=True)
    return report


def aggregate_qa(units, manifest):
    bundles = {}
    for method in METHODS:
        records = [u["record"] for u in units if u["method"] == method and u.get("record") is not None]
        if not records:
            bundles[method] = None
            continue
        run = deepcopy(next(u["bundle_run"] for u in units if u["method"] == method and u.get("record") is not None))
        run.update(id=manifest["experiment_id"] + "__" + method, experiment_id=manifest["experiment_id"],
                   finished_at=now(), experiment_config=manifest["config"],
                   quality_scope="pending_human_review")
        bundle = {"schema_version": 1, "run": run, "records": sorted(records, key=lambda r: r["id"])}
        bundle["capture_sha256"] = capture_digest(bundle)
        validate_qa_results(bundle)
        bundles[method] = bundle
    return {"schema_version": 1, "experiment_id": manifest["experiment_id"], "config": manifest["config"],
            "initialization": manifest["initialization"], "planned_units": 200, "finished_units": len(units),
            "completed_at": now(), "summary": summarize_units(units), "methods": bundles, "units": units,
            "human_review": {"status": "pending", "reviewed": 0, "quality_metrics": None},
            "cost": {"currency": None, "amount": None, "reason": "No verified per-request billing or price snapshot; report returned tokens only"}}


def qa_text(report):
    lines = ["V3 real QA comparison", "experiment_id=" + report["experiment_id"],
             "Manual quality review pending; no automatic quality scores.",
             json.dumps(report["summary"], ensure_ascii=False, indent=2)]
    for unit in report["units"]:
        lines.extend(["", unit["unit_id"] + " status=" + unit["status"], "question: " + unit["question"]])
        record = unit.get("record")
        if record:
            lines.extend(["sources: " + ", ".join(record["context_document_ids"]), "answer: " + record["answer"]])
        else:
            lines.append("failure: " + str(unit.get("failed_stage")) + "/" + str(unit.get("exception_type")))
    return "\n".join(lines) + "\n"


def publish_review_bundles(report, output):
    for method, bundle in report["methods"].items():
        if bundle is None:
            continue
        path = output / "v3_qa_review" / (method + ".json")
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_qa_results(existing)
            if existing["capture_sha256"] != bundle["capture_sha256"]:
                raise ValueError("Existing review capture differs")
            continue  # 人工 review 即使已填写，也不覆盖。
        validate_qa_results(bundle)
        save_report(bundle, path)


def inherit_windows_user_environment(arguments):
    """启动新进程时继承 Windows 已保存的用户环境；不调用 SetEnvironmentVariable。"""
    import ctypes
    from ctypes import wintypes
    import subprocess
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    userenv.CreateEnvironmentBlock.argtypes = [ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.BOOL]
    userenv.DestroyEnvironmentBlock.argtypes = [ctypes.c_void_p]
    token, block = wintypes.HANDLE(), ctypes.c_void_p()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008 | 0x0002, ctypes.byref(token)):
        raise RuntimeError("User environment token unavailable")
    try:
        if not userenv.CreateEnvironmentBlock(ctypes.byref(block), token, False):
            raise RuntimeError("User environment unavailable")
        try:
            environment, address = {}, block.value
            while True:
                item = ctypes.wstring_at(address)
                if not item:
                    break
                address += (len(item) + 1) * ctypes.sizeof(ctypes.c_wchar)
                if not item.startswith("="):
                    name, value = item.split("=", 1)
                    environment[name] = value
            return subprocess.call([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), *arguments],
                                   env=environment, cwd=ROOT)
        finally:
            userenv.DestroyEnvironmentBlock(block)
    finally:
        kernel.CloseHandle(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--user-environment", action="store_true", help="Windows: inherit the saved user environment in a child process without editing it")
    args = parser.parse_args()
    if args.user_environment:
        return inherit_windows_user_environment([a for a in sys.argv[1:] if a != "--user-environment"])
    logging.disable(logging.CRITICAL)  # 输出仅为受控进度；原始答案只写明确授权的采集文件。
    corpus = load_dataset(ROOT / "data/evaluation/retrieval_eval_v3.json")
    qa = load_qa_dataset(ROOT / "data/evaluation/qa_eval_v3.json", corpus)
    settings = Settings.from_env()
    provider = validate_provider(settings)
    config = {**provider, "retrieval_sha256": corpus.sha256, "qa_sha256": qa.sha256,
              "candidate_k": 10, "context_top_k": 3, "ks": [3, 5, 10], "alpha": .7,
              "embedding_model": "BAAI/bge-small-zh-v1.5", "reranker_model": "BAAI/bge-reranker-base",
              "prompt_source_sha256": hashlib.sha256((ROOT / "app/generation/prompt.py").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "request_parameters": "Existing OpenAILLM sends model and input only; no additional generation parameters",
              "query_id_definition": "QA query_id equals qa_id; retrieval_eval_query_ids contains exact question matches only"}
    output = ROOT / "results"
    manifest_path = output / "v3_experiment_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume to reuse existing paid-unit progress")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Experiment config or code fingerprint changed; refusing ambiguous resume")
    else:
        for name in ("v3_retrieval_comparison.json", "v3_retrieval_comparison.txt", "v3_qa_comparison.json", "v3_qa_comparison.txt"):
            if (output / name).exists():
                raise FileExistsError("Requested output already exists")
        manifest = {"experiment_id": uuid4().hex, "started_at": now(), "config": config}
    if (output / "v3_qa_comparison.json").exists():
        completed = json.loads((output / "v3_qa_comparison.json").read_text(encoding="utf-8"))
        if completed["experiment_id"] != manifest["experiment_id"]:
            raise ValueError("Existing QA report identity mismatch")
        publish_text(qa_text(completed), output / "v3_qa_comparison.txt")
        publish_review_bundles(completed, output)
        complete = completed["finished_units"] == completed["planned_units"] and not completed.get("stopped_early", False)
        print(json.dumps({"event": "final_result_already_exists", "action": "No calls made", "complete": complete}), flush=True)
        return 0 if complete else 2
    print(json.dumps({"event": "configuration_verified", "provider": provider, "experiment_id": manifest["experiment_id"]}), flush=True)
    retrievers, reranker, initialization = build_models(corpus)
    if not manifest_path.exists():
        manifest["initialization"] = initialization
        save_report(manifest, manifest_path)
    print(json.dumps({"event": "models_loaded_from_cache", "initialization": initialization}), flush=True)
    evaluate_retrieval(corpus, retrievers, reranker, manifest, output)
    store = ProgressStore(output / "v3_qa_progress", manifest["experiment_id"])
    llm = OpenAILLM(settings=settings)
    monitor = LLMMonitor(llm, store)
    samples = [qa.samples[0], qa.samples[3]] + [s for s in qa.samples if s["id"] not in {"qa_001", "qa_004"}]
    units, stopped = [], False
    try:
        for index, sample in enumerate(samples):
            order = METHODS[index % 4:] + METHODS[:index % 4]
            for method in order:
                unit = run_cell(method, sample, corpus, qa, retrievers[method],
                                reranker if method == "hybrid_reranker" else None, llm, monitor, store)
                units.append(unit)
                print(json.dumps({"event": "qa_unit", "done": len(units), "unit": unit["unit_id"],
                                  "status": unit["status"], "http_requests": len(unit["http_attempts"]),
                                  "total_ms": round(unit["timings_ms"]["total"], 2)}), flush=True)
                if should_stop(units):
                    stopped = True
                    break
            if index == 1:
                if any(u["status"] != "success" for u in units[:8]):
                    stopped = True
                else:
                    print(json.dumps({"event": "smoke_passed", "reused_formal_units": 8}), flush=True)
            if stopped:
                break
        report = aggregate_qa(units, manifest)
        report["stopped_early"] = stopped
        report["sdk_policy"] = {"max_retries": llm.client.max_retries, "timeout_seconds": 60,
                                "batch_level_retries": 0, "manual_replay_of_failed_or_inflight_units": False}
        save_report(report, output / "v3_qa_comparison.json")
        publish_text(qa_text(report), output / "v3_qa_comparison.txt")
        publish_review_bundles(report, output)
        print(json.dumps({"event": "qa_completed", "finished_units": len(units), "stopped_early": stopped,
                          "summary": report["summary"]}), flush=True)
        return 0 if not stopped else 2
    finally:
        llm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"event": "experiment_interrupted", "action": "Inspect in_flight progress before resuming"}), flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({"event": "experiment_failed", "exception_type": safe_type(error)}), flush=True)
        raise SystemExit(1)
