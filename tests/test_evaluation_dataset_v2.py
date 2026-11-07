"""V2 固定数据契约：仅检查人工裁决、版本绑定和纯指标，不运行检索。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.qa_dataset import load_qa_dataset
from app.evaluation.retrieval_metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    validate_relevance,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
FROZEN_SHA256 = {
    "data/evaluation/qa_eval.json": "9e33ac59dd0ba406a047249442672a5a01e22ef73f8f4d3c4eb054133d89c1eb",
    "data/evaluation/retrieval_eval.json": "085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71",
    "results/day7_retrieval.json": "8c49a887f05ad146311b84bff3dc83fd952baf18c4e9d059b985dfc4d172e2e6",
    "results/day7_retrieval.txt": "b3e02ebf24067f329907cc5a4232779f521f72ba74249614a3f4650126569445",
    "results/day8_qa_collect.txt": "6d79999416374e27e42c7d5c7a2c09968f29e7c3f7bdb707ed9a2cd3c6c2af3b",
    "results/day8_qa_deepseek_ai_review_v1.json": "21c22fe324a06d549d1ab10b8edad618e202f5fc60916e1af7770a79ca0ef2e7",
    "results/day8_qa_deepseek_reviewed_summary_v1.json": "fad7ae432f75f0d2106f6ec57f1414dc8af2dd3e570aaa0da2ff9e5f2bfae936",
    "results/day8_qa_deepseek_reviewed_v1.before_qa001_2329d1e4.json": "5e912e10ca821b2818b8c762b42b8246e781e9ecfa52a0fb045608e25a444e8c",
    "results/day8_qa_deepseek_reviewed_v1.json": "1b95fa47608ddca85d9ca7c45b1d901d45ab645e7dfa00f1f8cd23baa974ea29",
    "results/day8_qa_deepseek_summary_v1.json": "64d10a1a15101d085082cda2c2b7778dbd15b276d832ff8b5b371c8e7df5527e",
    "results/day8_qa_deepseek_v1.json": "333899db1b3e45b35335c0b4fed7e282b3181205921ccdd5530a0276af84b57f",
    "results/day8_qa_mock.json": "2468464084e9755e9a9de21c65419dea39fe3119627ae012e5b183f8ace6b48d",
    "results/day8_qa_summary.json": "fa3ec4ccfb3eef4398219f0952e91397a2ebbe5d7b29f2956cba9a1f1bf73ecb",
}


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path, expected", FROZEN_SHA256.items())
def test_original_datasets_and_all_day7_day8_history_are_unchanged(relative_path, expected):
    assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_v2_keeps_all_documents_and_validates_existing_id_mapping():
    original = load_dataset(DATA / "retrieval_eval.json")
    revised = load_dataset(DATA / "retrieval_eval_v2.json")
    assert len(revised.documents) == 18
    assert len(revised.queries) == 14
    assert revised.documents == original.documents
    assert len({d.metadata["eval_id"] for d in revised.documents}) == 18
    assert len({q["id"] for q in revised.queries}) == 14
    known_ids = {d.metadata["eval_id"] for d in revised.documents}
    for query in revised.queries:
        labels = {item["id"]: item["relevance"] for item in query["relevant_documents"]}
        validate_relevance(labels)
        assert set(labels) <= known_ids
        assert len(labels) == len(query["relevant_documents"])


def test_v2_has_exactly_the_two_adjudicated_label_changes():
    original = read_json("retrieval_eval.json")
    revised = read_json("retrieval_eval_v2.json")
    expected_queries = deepcopy(original["queries"])
    queries = {query["id"]: query for query in expected_queries}
    next(item for item in queries["q04"]["relevant_documents"] if item["id"] == "cuda_oom")["relevance"] = 1
    queries["q13"]["relevant_documents"].append({"id": "rag_overview", "relevance": 1})
    assert revised["queries"] == expected_queries
    original_q02 = next(q for q in original["queries"] if q["id"] == "q02")
    revised_q02 = next(q for q in revised["queries"] if q["id"] == "q02")
    assert revised_q02 == original_q02
    assert next(item["relevance"] for item in revised_q02["relevant_documents"] if item["id"] == "embedding_bge") == 1


def test_retrieval_v2_changes_only_allowed_metadata_and_queries():
    original = read_json("retrieval_eval.json")
    revised = read_json("retrieval_eval_v2.json")
    assert revised.keys() == original.keys()
    for key in original.keys() - {"metadata", "queries"}:
        assert revised[key] == original[key]
    assert revised["metadata"].keys() == original["metadata"].keys()
    changed_keys = {
        key for key in original["metadata"]
        if original["metadata"][key] != revised["metadata"][key]
    }
    assert changed_keys == {"name", "version", "annotation_status", "annotation_process"}
    assert revised["metadata"]["version"] == 2
    assert revised["metadata"]["name"] == "rag_knowledge_teaching_v2"
    assert revised["metadata"]["annotation_status"] == "teaching_labels_partially_human_adjudicated_remaining_pending"
    process = revised["metadata"]["annotation_process"]
    for pair in ("q02/embedding_bge", "q04/cuda_oom", "q13/rag_overview"):
        assert pair in process
    assert "其余历史标签仍待用户人工复核" in process


@pytest.mark.parametrize("filename", ["retrieval_eval.json", "retrieval_eval_v2.json"])
def test_retrieval_fingerprint_matches_actual_file_bytes(filename):
    dataset = load_dataset(DATA / filename)
    assert dataset.sha256 == hashlib.sha256((DATA / filename).read_bytes()).hexdigest()


def test_qa_v2_only_changes_version_and_corpus_binding_metadata():
    original = read_json("qa_eval.json")
    revised = read_json("qa_eval_v2.json")
    expected = deepcopy(original)
    expected["version"] = "2.0"
    expected["corpus_sha256"] = hashlib.sha256((DATA / "retrieval_eval_v2.json").read_bytes()).hexdigest()
    expected["metadata"]["name"] = "rag_qa_teaching_v2"
    expected["metadata"]["corpus_file"] = "retrieval_eval_v2.json"
    assert revised == expected
    assert len(revised["samples"]) == 5
    assert revised["samples"] == original["samples"]
    assert revised["metadata"]["annotation_status"] == original["metadata"]["annotation_status"]


@pytest.mark.parametrize(
    "qa_name, retrieval_name, version",
    [
        ("qa_eval.json", "retrieval_eval.json", "1.0"),
        ("qa_eval_v2.json", "retrieval_eval_v2.json", "2.0"),
    ],
)
def test_matching_qa_and_retrieval_versions_load(qa_name, retrieval_name, version):
    corpus = load_dataset(DATA / retrieval_name)
    dataset = load_qa_dataset(DATA / qa_name, corpus)
    assert dataset.version == version
    assert dataset.corpus_sha256 == corpus.sha256
    assert dataset.sha256 == hashlib.sha256((DATA / qa_name).read_bytes()).hexdigest()
    assert len(dataset.samples) == 5


@pytest.mark.parametrize(
    "qa_name, retrieval_name",
    [
        ("qa_eval.json", "retrieval_eval_v2.json"),
        ("qa_eval_v2.json", "retrieval_eval.json"),
    ],
)
def test_cross_version_qa_binding_is_still_rejected(qa_name, retrieval_name):
    with pytest.raises(ValueError, match="语料 SHA256 不一致"):
        load_qa_dataset(DATA / qa_name, load_dataset(DATA / retrieval_name))


def test_deliberately_invalid_qa_fingerprint_is_rejected(tmp_path):
    raw = read_json("qa_eval_v2.json")
    raw["corpus_sha256"] = "0" * 64
    path = tmp_path / "invalid_qa.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="语料 SHA256 不一致"):
        load_qa_dataset(path, load_dataset(DATA / "retrieval_eval_v2.json"))


@pytest.mark.parametrize("change", ["label", "whitespace"])
def test_binding_covers_labels_and_whitespace_even_with_identical_documents(tmp_path, change):
    original_bytes = (DATA / "retrieval_eval_v2.json").read_bytes()
    if change == "label":
        raw = json.loads(original_bytes)
        next(item for item in raw["queries"][0]["relevant_documents"] if item["id"] == "rag_evidence")["relevance"] = 2
        changed_bytes = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    else:
        changed_bytes = original_bytes + b"\n"
    path = tmp_path / "changed_retrieval.json"
    path.write_bytes(changed_bytes)
    corpus = load_dataset(path)
    original = load_dataset(DATA / "retrieval_eval_v2.json")
    assert corpus.documents == original.documents
    assert corpus.sha256 != original.sha256
    with pytest.raises(ValueError, match="语料 SHA256 不一致"):
        load_qa_dataset(DATA / "qa_eval_v2.json", corpus)


def test_v2_weak_positive_labels_use_existing_metric_rules_on_fixed_rankings():
    # 手工指定排名，只算纯指标；不调用任何 Retriever 或四方案评估。
    corpus = load_dataset(DATA / "retrieval_eval_v2.json")
    labels = {
        query["id"]: {item["id"]: item["relevance"] for item in query["relevant_documents"]}
        for query in corpus.queries
    }
    q04_ranking = ["cuda_oom", "cuda_1176", "cuda_1175"]
    q13_ranking = ["rag_overview", "rag_limits", "rag_evidence"]
    assert recall_at_k(q04_ranking, labels["q04"], 1) == 0.5
    assert recall_at_k(q04_ranking, labels["q04"], 3) == 1
    assert reciprocal_rank(q04_ranking, labels["q04"]) == 1
    assert dcg_at_k([1, 0, 3], 3) == 4.5
    assert ndcg_at_k(q04_ranking, labels["q04"], 3) == pytest.approx(4.5 / (7 + 1 / math.log2(3)))
    assert recall_at_k(q13_ranking, labels["q13"], 1) == pytest.approx(1 / 3)
    assert ndcg_at_k(q13_ranking, labels["q13"], 3) == pytest.approx(
        (1 + 7 / math.log2(3) + 3 / 2) / (7 + 3 / math.log2(3) + 1 / 2)
    )
    assert mean_reciprocal_rank(
        [q04_ranking, ["embedding_bge", "rag_overview"]],
        [labels["q04"], labels["q13"]],
    ) == 0.75
