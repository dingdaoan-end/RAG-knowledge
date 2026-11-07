"""连接恢复后仅执行未开始单元；绝不重放已完成或失败的付费请求。"""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v3_comprehensive import (
    METHODS, InFlightError, LLMMonitor, OpenAILLM, ProgressStore, Settings,
    aggregate_qa, build_models, load_dataset, load_qa_dataset, now,
    qa_text, run_cell, safe_type, save_report, should_stop, validate_provider, capture_digest,
)


def run_missing(schedule, store, execute):
    units, created = [], []
    stopped = False
    for method, sample in schedule:
        unit_id = method + "__" + sample["id"]
        if store.path(unit_id).exists():
            unit = store.load(unit_id)
            if unit["state"] != "finished":
                raise InFlightError("Uncertain paid unit requires inspection, not replay")
        else:
            unit = execute(method, sample)
            created.append(unit)
            print(json.dumps({"event": "continued_unit", "new_units": len(created), "unit": unit_id,
                              "status": unit["status"], "http_requests": len(unit.get("http_attempts", []))}), flush=True)
            # 恢复后的首个正式单元亦作为连通性验证；失败则立即停，不另发 Smoke 请求。
            if (len(created) == 1 and unit["status"] != "success") or should_stop(created):
                stopped = True
        units.append(unit)
        if stopped:
            break
    return units, created, stopped


def preserve_and_replace(path, content, snapshot):
    """仅接替本轮活动汇总；旧字节先以新名字完整保留，原始 checkpoint 不改。"""
    path, snapshot = Path(path), Path(snapshot)
    old = path.read_bytes() if path.exists() else None
    if old is not None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != old:
            snapshot = snapshot.with_name(snapshot.stem + "." + hashlib.sha256(old).hexdigest()[:16] + snapshot.suffix)
        if snapshot.exists() and snapshot.read_bytes() != old:
            raise ValueError("Interrupted snapshot bytes differ; refusing replacement")
        if not snapshot.exists():
            with snapshot.open("xb") as handle:
                handle.write(old)
                handle.flush()
                os.fsync(handle.fileno())
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".continued-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if old is not None and path.read_bytes() != old:
            raise ValueError("Output changed concurrently")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    logging.disable(logging.CRITICAL)
    output = ROOT / "results"
    manifest = json.loads((output / "v3_experiment_manifest.json").read_text(encoding="utf-8"))
    old = json.loads((output / "v3_qa_comparison.json").read_text(encoding="utf-8"))
    if old["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Experiment identity mismatch")
    if old["finished_units"] == 200:
        print(json.dumps({"event": "already_complete", "new_requests": 0}), flush=True)
        return 0
    config = manifest["config"]
    original_runner = ROOT / "scripts/run_v3_comprehensive.py"
    if hashlib.sha256(original_runner.read_bytes()).hexdigest() != config["runner_sha256"]:
        raise ValueError("Original runner changed; do not mix experiment implementations")
    if hashlib.sha256((ROOT / "app/generation/prompt.py").read_bytes()).hexdigest() != config["prompt_source_sha256"]:
        raise ValueError("Prompt source changed; do not mix generation conditions")
    settings = Settings.from_env()
    provider = validate_provider(settings)
    if any(provider[key] != config[key] for key in provider):
        raise ValueError("Provider configuration changed")
    corpus = load_dataset(ROOT / "data/evaluation/retrieval_eval_v3.json")
    qa = load_qa_dataset(ROOT / "data/evaluation/qa_eval_v3.json", corpus)
    if corpus.sha256 != config["retrieval_sha256"] or qa.sha256 != config["qa_sha256"]:
        raise ValueError("Dataset fingerprints changed")
    store = ProgressStore(output / "v3_qa_progress", manifest["experiment_id"])
    for unit in old["units"]:
        if store.load(unit["unit_id"]) != unit:
            raise ValueError("Existing unit changed")
    # 本轮人工评分仍应为空；不接替任何已由用户填写的评审文件。
    for path in (output / "v3_qa_review").glob("*.json"):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle["run"]["experiment_id"] != manifest["experiment_id"] or any(
            record["review"]["status"] != "pending" or record["review"]["review_notes"]
            for record in bundle["records"]
        ):
            raise ValueError("Human review exists; preserve it and choose a new review output")
    retrievers, reranker, initialization = build_models(corpus)
    if initialization["models"] != manifest["initialization"]["models"]:
        raise ValueError("Cached model revisions changed")
    llm = OpenAILLM(settings=settings)
    monitor = LLMMonitor(llm, store)
    samples = [qa.samples[0], qa.samples[3]] + [s for s in qa.samples if s["id"] not in {"qa_001", "qa_004"}]
    schedule = [(method, sample) for index, sample in enumerate(samples)
                for method in METHODS[index % 4:] + METHODS[:index % 4]]
    started = now()
    try:
        units, created, stopped = run_missing(schedule, store, lambda method, sample: run_cell(
            method, sample, corpus, qa, retrievers[method], reranker if method == "hybrid_reranker" else None,
            llm, monitor, store))
    finally:
        llm.close()
    report = aggregate_qa(units, manifest)
    for method, bundle in report["methods"].items():
        if bundle:
            bundle["run"]["elapsed_seconds"] = sum(u["timings_ms"]["total"] for u in units if u["method"] == method) / 1000
            bundle["run"]["elapsed_scope"] = "sum_of_method_unit_durations_including_failed_units"
            bundle["capture_sha256"] = capture_digest(bundle)
    report["sdk_policy"] = old["sdk_policy"]
    report["stopped_early"] = stopped
    report["continuation"] = {"started_at": started, "finished_at": now(), "old_units_reused": len(units) - len(created),
                              "new_units": len(created), "old_failed_units_retried": 0,
                              "initialization": initialization,
                              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "reason": "TCP connectivity restored after three APIConnectionError units; only unstarted cells authorized"}
    snapshot = output / "v3_interrupted_snapshot"
    def encode(value):
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # checkpoint 永不改写；只更新本轮活动汇总，所有中止版字节均先保留。
    for method, bundle in report["methods"].items():
        if bundle:
            preserve_and_replace(output / "v3_qa_review" / (method + ".json"), encode(bundle), snapshot / "review" / (method + ".json"))
    preserve_and_replace(output / "v3_qa_comparison.txt", qa_text(report).encode("utf-8"), snapshot / "v3_qa_comparison.txt")
    preserve_and_replace(output / "v3_qa_comparison.json", encode(report), snapshot / "v3_qa_comparison.json")
    print(json.dumps({"event": "continuation_completed", "finished_units": len(units), "new_units": len(created),
                      "requests": sum(len(u["http_attempts"]) for u in units), "stopped_early": stopped}), flush=True)
    return 0 if not stopped and len(units) == 200 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"event": "continuation_interrupted", "action": "Inspect in_flight checkpoint"}), flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({"event": "continuation_failed", "exception_type": safe_type(error)}), flush=True)
        raise SystemExit(1)
