import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def raw_dataset():
    return {
        "metadata": {"name": "unit", "annotation_status": "test_fixture"},
        "documents": [
            {"id": key, "text": text, "source": f"{key}.md", "chunk_id": 0}
            for key, text in [("a", "RAG 原理"), ("b", "RAG 工程"), ("c", "足球"), ("d", "烹饪")]
        ],
        "queries": [{"id": "q1", "query": "RAG", "relevant_documents": [{"id": "a", "relevance": 3}, {"id": "b", "relevance": 1}]}],
    }


def write_dataset(tmp_path, raw):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def test_dataset_ids_are_stable_across_loads_and_local_chunk_ids_can_repeat(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset

    path = write_dataset(tmp_path, raw_dataset)
    first, second = load_dataset(path), load_dataset(path)
    assert [document.metadata["eval_id"] for document in first.documents] == ["a", "b", "c", "d"]
    assert first.documents[0] is not second.documents[0]
    assert first.sha256 == second.sha256
    assert all(document.metadata["chunk_id"] == 0 for document in first.documents)


@pytest.mark.parametrize("bad_case", ["duplicate_document", "empty_document_id", "missing_document_id", "duplicate_query", "blank_query", "duplicate_label", "unknown_label", "no_positive", "grade_out_of_range", "bool_grade", "empty_documents", "empty_queries", "blank_text", "bad_chunk_id", "duplicate_source_chunk"])
def test_invalid_dataset_is_rejected(tmp_path, raw_dataset, bad_case):
    from app.evaluation.dataset import load_dataset

    documents, queries = raw_dataset["documents"], raw_dataset["queries"]
    if bad_case == "duplicate_document":
        documents[1]["id"] = "a"
    elif bad_case == "empty_document_id":
        documents[0]["id"] = ""
    elif bad_case == "missing_document_id":
        del documents[0]["id"]
    elif bad_case == "duplicate_query":
        queries.append(dict(queries[0]))
    elif bad_case == "blank_query":
        queries[0]["query"] = " "
    elif bad_case == "duplicate_label":
        queries[0]["relevant_documents"].append({"id": "a", "relevance": 2})
    elif bad_case == "unknown_label":
        queries[0]["relevant_documents"][0]["id"] = "not-in-corpus"
    elif bad_case == "no_positive":
        queries[0]["relevant_documents"] = [{"id": "a", "relevance": 0}]
    elif bad_case == "grade_out_of_range":
        queries[0]["relevant_documents"][0]["relevance"] = 4
    elif bad_case == "bool_grade":
        queries[0]["relevant_documents"][0]["relevance"] = True
    elif bad_case == "empty_documents":
        raw_dataset["documents"] = []
    elif bad_case == "empty_queries":
        raw_dataset["queries"] = []
    elif bad_case == "blank_text":
        documents[0]["text"] = " "
    elif bad_case == "bad_chunk_id":
        documents[0]["chunk_id"] = -1
    elif bad_case == "duplicate_source_chunk":
        documents[1]["source"] = documents[0]["source"]
    with pytest.raises(ValueError):
        load_dataset(write_dataset(tmp_path, raw_dataset))


def build_methods(documents):
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever
    from app.retrieval.retriever import FaissRetriever
    from app.retrieval.text_retriever import TextRetriever
    from app.retrieval.reranker import CrossEncoderReranker

    class Embedder:
        def embed_query(self, query):
            return np.array([[1, 0]], dtype="float32")

    class Scorer:
        def __init__(self):
            self.pairs = []

        def predict(self, pairs, **kwargs):
            self.pairs.extend(pairs)
            return [0.1, 0.9]

    scorer = Scorer()
    vector = TextRetriever(FaissRetriever(documents, np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype="float32")), Embedder())
    bm25 = BM25Retriever(documents)
    return vector, bm25, HybridRetriever(vector, bm25), CrossEncoderReranker(scorer=scorer), scorer


def test_four_methods_share_ground_truth_and_reranker_only_sees_hybrid_candidates(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import evaluate_methods, format_report, save_report

    dataset = load_dataset(write_dataset(tmp_path, raw_dataset))
    vector, bm25, hybrid, reranker, scorer = build_methods(dataset.documents)
    report = evaluate_methods(dataset, vector, bm25, hybrid, reranker, ks=[1, 2], candidate_k=2)
    assert set(report["summary"]) == {"vector", "bm25", "hybrid", "hybrid_reranker"}
    assert report["summary"]["hybrid"]["Recall@1"] == 0.5
    assert report["summary"]["hybrid"]["nDCG@1"] == 1
    assert report["summary"]["hybrid_reranker"]["nDCG@1"] == pytest.approx(1 / 7)
    assert report["summary"]["hybrid_reranker"]["Recall@2"] == 1
    assert report["summary"]["hybrid_reranker"]["MRR@2"] == 1
    methods = report["queries"][0]["methods"]
    assert [row["id"] for row in methods["hybrid_reranker"]["ranking"]] == ["b", "a"]
    assert methods["hybrid_reranker"]["candidate_count"] == 2
    assert methods["hybrid_reranker"]["first_relevant_rank"] == 1
    assert scorer.pairs == [("RAG", "RAG 原理"), ("RAG", "RAG 工程")]
    assert report["analysis"]["reranker_changed_queries"] == ["q1"]
    assert report["analysis"]["reranker_ndcg_delta"]["nDCG@1"] < 0
    path = tmp_path / "results" / "report.json"
    save_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert "MRR@2" in format_report(report) and "q1" in format_report(report)


def test_unretrieved_relevant_document_is_not_added_by_reranker(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import evaluate_methods

    raw_dataset["queries"][0]["relevant_documents"] = [{"id": "c", "relevance": 3}]
    dataset = load_dataset(write_dataset(tmp_path, raw_dataset))
    vector, bm25, hybrid, reranker, scorer = build_methods(dataset.documents)
    report = evaluate_methods(dataset, vector, bm25, hybrid, reranker, ks=[1, 2], candidate_k=2)
    assert report["summary"]["hybrid_reranker"]["MRR@2"] == 0
    assert report["queries"][0]["methods"]["hybrid_reranker"]["first_relevant_rank"] is None
    assert report["analysis"]["no_relevant_hits"]["hybrid_reranker"] == ["q1"]


@pytest.mark.parametrize("bad_case", ["missing_id", "unknown_id", "duplicate_result", "changed_text", "foreign_candidate", "partial_rerank"])
def test_invalid_result_mapping_or_rerank_candidates_fail_explicitly(tmp_path, raw_dataset, bad_case):
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import evaluate_methods
    from app.ingestion.document import Document
    from app.retrieval.retriever import SearchResult

    dataset = load_dataset(write_dataset(tmp_path, raw_dataset))
    vector, bm25, hybrid, reranker, _ = build_methods(dataset.documents)
    if bad_case in {"missing_id", "unknown_id", "changed_text"}:
        document = Document("RAG 原理", dict(dataset.documents[0].metadata))
        if bad_case == "missing_id":
            del document.metadata["eval_id"]
        elif bad_case == "unknown_id":
            document.metadata["eval_id"] = "unknown"
        else:
            document.text = "wrong chunk"
        vector.search = lambda *args, **kwargs: [SearchResult(document, 1, 1)]
    elif bad_case == "duplicate_result":
        result = SearchResult(dataset.documents[0], 1, 1)
        vector.search = lambda *args, **kwargs: [result, result]
    elif bad_case == "foreign_candidate":
        reranker.rerank = lambda *args, **kwargs: [SearchResult(dataset.documents[2], 1, 1)]
    elif bad_case == "partial_rerank":
        original = reranker.rerank
        reranker.rerank = lambda query, candidates, **kwargs: original(query, candidates, top_k=1)
    with pytest.raises(ValueError):
        evaluate_methods(dataset, vector, bm25, hybrid, reranker, ks=[1, 2], candidate_k=2)


def test_empty_results_are_reported_without_padding_or_model_call(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import evaluate_methods

    dataset = load_dataset(write_dataset(tmp_path, raw_dataset))

    class EmptyRetriever:
        def search(self, query, top_k):
            return []

    class NeverRerank:
        def rerank(self, *args, **kwargs):
            pytest.fail("空候选不应调用重排模型")

    empty = EmptyRetriever()
    report = evaluate_methods(dataset, empty, empty, empty, NeverRerank(), ks=[1, 2], candidate_k=2)
    assert all(metrics["MRR@2"] == 0 for metrics in report["summary"].values())
    assert all(row["candidate_count"] == 0 for row in report["queries"][0]["methods"].values())


def test_candidate_budget_must_cover_reported_k(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import evaluate_methods

    dataset = load_dataset(write_dataset(tmp_path, raw_dataset))
    vector, bm25, hybrid, reranker, _ = build_methods(dataset.documents)
    with pytest.raises(ValueError):
        evaluate_methods(dataset, vector, bm25, hybrid, reranker, ks=[3], candidate_k=2)


def test_cli_help_works_outside_project_directory_without_loading_models(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout and "--candidate-k" in result.stdout


def test_duplicate_json_keys_are_rejected_instead_of_overwriting_ids(tmp_path, raw_dataset):
    from app.evaluation.dataset import load_dataset

    path = tmp_path / "duplicate.json"
    path.write_text('{"documents": [], ' + json.dumps(raw_dataset)[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.parametrize("policy", [{"unlisted_relevance": 2}, {"binary_relevance_threshold": 2}])
def test_dataset_cannot_silently_change_metric_policy(tmp_path, raw_dataset, policy):
    from app.evaluation.dataset import load_dataset

    raw_dataset["metadata"].update(policy)
    with pytest.raises(ValueError):
        load_dataset(write_dataset(tmp_path, raw_dataset))


def test_cli_refuses_to_overwrite_ground_truth(tmp_path, raw_dataset, monkeypatch):
    import runpy

    path = write_dataset(tmp_path, raw_dataset)
    before = path.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "load_dataset", lambda *args: pytest.fail("覆盖保护必须先于数据及模型加载"))
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(path), "--output", str(path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert path.read_bytes() == before


def test_save_report_refuses_existing_file_and_can_save_new_path(tmp_path):
    from app.evaluation.evaluator import save_report

    path = tmp_path / "day7_retrieval.json"
    original = b'{"frozen": true}\r\n'
    path.write_bytes(original)
    with pytest.raises(FileExistsError, match="新的输出路径"):
        save_report({"new": "报告"}, path)
    assert path.read_bytes() == original
    new_path = tmp_path / "new_experiment" / "report.json"
    save_report({"new": "报告"}, new_path)
    assert json.loads(new_path.read_text(encoding="utf-8")) == {"new": "报告"}
    assert list(new_path.parent.iterdir()) == [new_path]


def test_serialization_failure_leaves_no_report_or_temporary_file(tmp_path):
    from app.evaluation.evaluator import save_report

    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        save_report({"valid_prefix": "报告", "invalid": float("nan")}, path)
    assert list(tmp_path.iterdir()) == []


def test_partial_write_failure_leaves_no_successful_report(tmp_path, monkeypatch):
    import app.evaluation.evaluator as evaluator

    def failed_dump(report, handle, **kwargs):
        handle.write('{"partial":')
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(evaluator.json, "dump", failed_dump)
    path = tmp_path / "report.json"
    with pytest.raises(OSError, match="disk write failure"):
        evaluator.save_report({"report": "完整报告"}, path)
    assert list(tmp_path.iterdir()) == []


def test_report_created_during_save_is_not_overwritten(tmp_path, monkeypatch):
    import os
    from app.evaluation.evaluator import save_report

    path = tmp_path / "report.json"
    original_link = os.link
    concurrent_report = b'{"other_experiment": true}\n'

    def competing_link(source, target):
        Path(target).write_bytes(concurrent_report)
        original_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError, match="新的输出路径"):
        save_report({"this_experiment": True}, path)
    assert path.read_bytes() == concurrent_report
    assert list(tmp_path.iterdir()) == [path]


def test_publication_failure_leaves_no_report_or_temporary_file(tmp_path, monkeypatch):
    import os
    from app.evaluation.evaluator import save_report

    def failed_link(source, target):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", failed_link)
    with pytest.raises(OSError, match="publication failure"):
        save_report({"complete": True}, tmp_path / "report.json")
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_existing_report_before_loading_models(tmp_path, raw_dataset, monkeypatch, capsys):
    import runpy

    path = tmp_path / "results" / "day7_retrieval.json"
    path.parent.mkdir()
    original = b'{"frozen": true}\n'
    path.write_bytes(original)
    dataset = write_dataset(tmp_path, raw_dataset)
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "load_dataset", lambda *args: pytest.fail("拒绝覆盖应先于模型加载"))
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(dataset), "--output", str(path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "新的输出路径" in capsys.readouterr().err
    assert path.read_bytes() == original


def test_failed_evaluation_does_not_change_existing_report(tmp_path, raw_dataset, monkeypatch):
    import runpy
    from types import ModuleType
    from app.retrieval.text_retriever import TextRetriever

    path = tmp_path / "day7_retrieval.json"
    original = b'{"frozen": true}\n'
    path.write_bytes(original)
    dataset = write_dataset(tmp_path, raw_dataset)
    output = tmp_path / "new_experiment.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
    main = runpy.run_path(str(script))["main"]
    # 不导入真实模型依赖；CLI、FAISS、评估函数和保存流程仍使用真实代码。
    class FakeEmbedder:
        def __init__(self, **kwargs):
            pass

        def embed_documents(self, docs):
            return np.eye(len(docs), dtype="float32")

    embedding_module = ModuleType("app.embedding.embedder")
    embedding_module.TextEmbedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "app.embedding.embedder", embedding_module)

    def fail(self, query, top_k):
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(TextRetriever, "search", fail)
    monkeypatch.setattr(sys, "argv", [str(script), "--dataset", str(dataset), "--output", str(output)])
    with pytest.raises(RuntimeError, match="retrieval failed"):
        main()
    assert path.read_bytes() == original
    assert not output.exists()
