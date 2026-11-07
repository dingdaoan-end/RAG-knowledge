"""冻结旧数据、严格 V3 字段、证据可定位及版本绑定；不运行模型或检索。"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.qa_dataset import load_qa_dataset
from app.evaluation.retrieval_metrics import validate_relevance


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
FROZEN = {
    "data/evaluation/qa_eval.json": "9e33ac59dd0ba406a047249442672a5a01e22ef73f8f4d3c4eb054133d89c1eb",
    "data/evaluation/qa_eval_v2.json": "fa24d5ca860ecf2c07365d662f5ff1cc5da00a2e86e9e67c5189c3a4752f3092",
    "data/evaluation/retrieval_eval.json": "085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71",
    "data/evaluation/retrieval_eval_v2.json": "d204aef8d060a86909f8a239e8bf93630ce375136db9efbf971c28539ecc458e",
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


def raw(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name, digest", FROZEN.items())
def test_v1_v2_and_historical_results_keep_original_bytes(name, digest):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_v3_retrieval_keeps_old_questions_and_documents_without_extra_fields():
    v2 = raw("retrieval_eval_v2.json")
    v3 = raw("retrieval_eval_v3.json")
    corpus = load_dataset(DATA / "retrieval_eval_v3.json")
    assert set(v3) == {"metadata", "documents", "queries"}
    assert v3["documents"] == v2["documents"]
    assert v3["queries"][:14] == v2["queries"]
    assert len(corpus.documents) == 18
    assert len(corpus.queries) == 50
    assert [q["id"] for q in corpus.queries] == [f"q{i:02d}" for i in range(1, 51)]
    for q in v3["queries"]:
        assert set(q) == {"id", "query", "relevant_documents"}
        for label in q["relevant_documents"]:
            assert set(label) == {"id", "relevance"}
        validate_relevance({item["id"]: item["relevance"] for item in q["relevant_documents"]})
    labels = {q["id"]: {x["id"]: x["relevance"] for x in q["relevant_documents"]} for q in v3["queries"]}
    assert labels["q02"]["embedding_bge"] == 1
    assert labels["q04"]["cuda_oom"] == 1
    assert labels["q13"]["rag_overview"] == 1


def test_v3_qa_keeps_old_samples_and_has_only_existing_sample_fields():
    v2 = raw("qa_eval_v2.json")
    v3 = raw("qa_eval_v3.json")
    dataset = load_qa_dataset(DATA / "qa_eval_v3.json", load_dataset(DATA / "retrieval_eval_v3.json"))
    assert set(v3) == {"version", "corpus_sha256", "metadata", "samples"}
    assert v3["samples"][:5] == v2["samples"]
    assert len(dataset.samples) == 50
    assert [q["id"] for q in dataset.samples] == [f"qa_{i:03d}" for i in range(1, 51)]
    for sample in dataset.samples:
        assert set(sample) == {"id", "question", "reference_answer", "required_facts", "relevant_document_ids", "answerable"}
    assert {s["id"] for s in dataset.samples if not s["answerable"]} == {
        "qa_004", "qa_005", "qa_045", "qa_046", "qa_047", "qa_048", "qa_049", "qa_050",
    }


@pytest.mark.parametrize("filename, field, text_key", [
    ("retrieval_eval_v3.json", "queries", "query"),
    ("qa_eval_v3.json", "samples", "question"),
])
def test_v3_questions_are_nonduplicate_after_punctuation_normalization(filename, field, text_key):
    questions = raw(filename)[field]
    normalized = [re.sub(r"[\W_]+", "", q[text_key]).casefold() for q in questions]
    assert len(set(normalized)) == 50


@pytest.mark.parametrize("filename", ["retrieval_eval_v3.json", "qa_eval_v3.json"])
def test_v3_has_no_review_markers(filename):
    def check(value):
        if isinstance(value, dict):
            assert not set(value) & {"pending", "verified", "review_status", "annotation_status", "annotation_process"}
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(raw(filename))


def test_new_answerable_qa_facts_have_literal_evidence_in_the_referenced_chunks():
    # 新事实使用原文短语；仅将冻结正文中路径的双反斜杠表示还原为标准路径。
    # 这种可定位性检查不代替对完整答案与等级的语义核查。
    corpus = load_dataset(DATA / "retrieval_eval_v3.json")
    qa = load_qa_dataset(DATA / "qa_eval_v3.json", corpus)
    texts = {doc.metadata["eval_id"]: doc.text for doc in corpus.documents}
    checked = 0
    for sample in qa.samples[5:]:
        if sample["answerable"]:
            checked += 1
            for fact in sample["required_facts"]:
                assert any(
                    fact in texts[doc_id].replace("\\\\", "\\")
                    for doc_id in sample["relevant_document_ids"]
                ), (sample["id"], fact)
    assert checked == 39


@pytest.mark.parametrize("query_id, document_id, grade", [
    ("q28", "pdf_ocr", 3),
    ("q29", "pdf_ocr", 1),
    ("q42", "rag_evidence", 1),
    ("q47", "cosine_unit", 1),
])
def test_reviewed_new_labels_distinguish_direct_answers_from_background(query_id, document_id, grade):
    query = next(q for q in raw("retrieval_eval_v3.json")["queries"] if q["id"] == query_id)
    assert next(label["relevance"] for label in query["relevant_documents"] if label["id"] == document_id) == grade


def test_reviewed_required_facts_include_the_requested_command_and_mechanisms():
    samples = {s["id"]: s for s in raw("qa_eval_v3.json")["samples"]}
    assert ".\\.venv\\Scripts\\python.exe main.py" in samples["qa_034"]["required_facts"]
    assert "减少句子和上下文在边界被截断的影响" in samples["qa_030"]["required_facts"]
    assert "分别做 Min-Max 归一化" in samples["qa_038"]["required_facts"]
    assert "alpha*vector_norm+(1-alpha)*bm25_norm" in samples["qa_038"]["required_facts"]


@pytest.mark.parametrize("suffix", ["", "_v2", "_v3"])
def test_each_qa_version_loads_only_with_its_byte_fingerprint(suffix, tmp_path):
    retrieval_path = DATA / f"retrieval_eval{suffix}.json"
    qa_path = DATA / f"qa_eval{suffix}.json"
    corpus = load_dataset(retrieval_path)
    dataset = load_qa_dataset(qa_path, corpus)
    assert corpus.sha256 == hashlib.sha256(retrieval_path.read_bytes()).hexdigest()
    assert dataset.corpus_sha256 == corpus.sha256
    assert dataset.sha256 == hashlib.sha256(qa_path.read_bytes()).hexdigest()
    wrong = json.loads(qa_path.read_text(encoding="utf-8"))
    wrong["corpus_sha256"] = "0" * 64
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(wrong, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="语料 SHA256 不一致"):
        load_qa_dataset(path, corpus)


@pytest.mark.parametrize("qa_suffix, corpus_suffix", [
    ("", "_v2"), ("", "_v3"), ("_v2", ""), ("_v2", "_v3"), ("_v3", ""), ("_v3", "_v2"),
])
def test_cross_version_pairings_are_rejected(qa_suffix, corpus_suffix):
    with pytest.raises(ValueError, match="语料 SHA256 不一致"):
        load_qa_dataset(DATA / f"qa_eval{qa_suffix}.json", load_dataset(DATA / f"retrieval_eval{corpus_suffix}.json"))
