from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from app.evaluation.dataset import load_dataset
from app.generation.llm import BaseLLM, MockLLM
from app.pipeline import RAGPipeline
from app.retrieval.retriever import SearchResult


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corpus():
    return load_dataset(ROOT / "data/evaluation/retrieval_eval.json")


@pytest.fixture
def qa_raw(corpus):
    return {
        "version": "1.0", "corpus_sha256": corpus.sha256,
        "metadata": {"annotation_status": "test_fixture"},
        "samples": [
            {"id": "qa_001", "question": "RAG 为什么减少幻觉？", "reference_answer": "人工参考答案，仅供评审，不得加入提示词。",
             "required_facts": ["检索外部知识", "不能保证完全消除幻觉"],
             "relevant_document_ids": ["rag_evidence", "rag_limits"], "answerable": True},
            {"id": "qa_002", "question": "这台机器的显卡具体型号是什么？", "reference_answer": "资料不足，不能确定当前硬件型号。",
             "required_facts": ["说明资料不足"], "relevant_document_ids": [], "answerable": False},
        ],
    }


def load_fixture(tmp_path, raw, corpus):
    from app.evaluation.qa_dataset import load_qa_dataset

    path = tmp_path / "qa.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return load_qa_dataset(path, corpus)


def make_pipeline(corpus, llm=None, reranker=None, top_k=1):
    documents = {document.metadata["eval_id"]: document for document in corpus.documents}

    class Retriever:
        def search(self, query):
            return [SearchResult(documents["rag_evidence"], 0.9, 1), SearchResult(documents["rag_limits"], 0.8, 2)]

    return RAGPipeline(Retriever(), llm or MockLLM(), reranker=reranker, top_k=top_k)


def test_qa_dataset_uses_real_stable_document_ids_and_records_negative_case(tmp_path, qa_raw, corpus):
    dataset = load_fixture(tmp_path, qa_raw, corpus)
    assert dataset.version == "1.0"
    assert dataset.corpus_sha256 == corpus.sha256
    assert dataset.samples[0]["relevant_document_ids"] == ["rag_evidence", "rag_limits"]
    assert dataset.samples[1]["answerable"] is False
    assert len(dataset.sha256) == 64


@pytest.mark.parametrize("field", ["id", "question", "reference_answer", "required_facts", "relevant_document_ids", "answerable"])
def test_missing_required_sample_fields_raise_value_error(tmp_path, qa_raw, corpus, field):
    del qa_raw["samples"][0][field]
    with pytest.raises(ValueError):
        load_fixture(tmp_path, qa_raw, corpus)


@pytest.mark.parametrize("bad_case", ["duplicate_id", "blank_id", "bad_id", "blank_question", "blank_reference", "facts_string", "facts_empty", "blank_fact", "duplicate_fact", "unknown_document", "duplicate_document", "ids_string", "bool_string", "answerable_without_evidence", "corpus_mismatch", "missing_version", "empty_samples"])
def test_qa_dataset_rejects_invalid_labels(tmp_path, qa_raw, corpus, bad_case):
    item = qa_raw["samples"][0]
    if bad_case == "duplicate_id":
        qa_raw["samples"][1]["id"] = item["id"]
    elif bad_case == "blank_id":
        item["id"] = " "
    elif bad_case == "bad_id":
        item["id"] = 1
    elif bad_case == "blank_question":
        item["question"] = " "
    elif bad_case == "blank_reference":
        item["reference_answer"] = ""
    elif bad_case == "facts_string":
        item["required_facts"] = "not a list"
    elif bad_case == "facts_empty":
        item["required_facts"] = []
    elif bad_case == "blank_fact":
        item["required_facts"] = [""]
    elif bad_case == "duplicate_fact":
        item["required_facts"] = ["重复", "重复"]
    elif bad_case == "unknown_document":
        item["relevant_document_ids"] = ["not_in_corpus"]
    elif bad_case == "duplicate_document":
        item["relevant_document_ids"] = ["rag_evidence", "rag_evidence"]
    elif bad_case == "ids_string":
        item["relevant_document_ids"] = "rag_evidence"
    elif bad_case == "bool_string":
        item["answerable"] = "true"
    elif bad_case == "answerable_without_evidence":
        item["relevant_document_ids"] = []
    elif bad_case == "corpus_mismatch":
        qa_raw["corpus_sha256"] = "0" * 64
    elif bad_case == "missing_version":
        del qa_raw["version"]
    elif bad_case == "empty_samples":
        qa_raw["samples"] = []
    with pytest.raises(ValueError):
        load_fixture(tmp_path, qa_raw, corpus)


def test_collection_captures_exact_prompt_context_sources_without_reference_leakage(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.generation.context_builder import build_context
    from app.generation.prompt import build_prompt

    class AnswerLLM(BaseLLM):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return "应检索外部知识，但不能保证完全消除幻觉。"

    llm = AnswerLLM()
    pipeline = make_pipeline(corpus, llm=llm)
    original_llm, original_retriever = pipeline.llm, pipeline.retriever
    dataset = load_fixture(tmp_path, qa_raw, corpus)
    bundle = collect_qa(pipeline, dataset, corpus)
    record = bundle["records"][0]
    expected_context = build_context(pipeline.retriever.search(record["question"])[:1])
    assert record["answer"] == "应检索外部知识，但不能保证完全消除幻觉。"
    assert record["context"] == expected_context
    assert record["prompt"] == llm.prompts[0] == build_prompt(record["question"], expected_context)
    assert record["context_document_ids"] == ["rag_evidence"]
    assert record["context_documents"][0]["metadata"]["eval_id"] == "rag_evidence"
    assert record["reference_answer"] not in record["prompt"]
    assert len(llm.prompts) == 2  # 每个样本只调用一次生成。
    assert pipeline.llm is original_llm and pipeline.retriever is original_retriever
    assert record["elapsed_seconds"] >= 0
    assert record["review"]["faithfulness"] is None
    assert record["review"]["unsupported_claims"] is None


@pytest.mark.parametrize("legacy", [False, True])
def test_sources_follow_actual_reranking_and_final_top_k(tmp_path, qa_raw, corpus, legacy):
    from app.evaluation.qa_evaluator import collect_qa

    class ReverseReranker:
        accepts_results = not legacy

        def rerank(self, query, candidates, top_k=3):
            return list(reversed(candidates))[:top_k]

    bundle = collect_qa(make_pipeline(corpus, reranker=ReverseReranker()), load_fixture(tmp_path, qa_raw, corpus), corpus)
    assert bundle["records"][0]["context_document_ids"] == ["rag_limits"]
    assert "rag_evidence" not in bundle["records"][0]["context_document_ids"]


def test_negative_sample_does_not_inject_reference_or_force_empty_context(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    record = bundle["records"][1]
    assert record["answerable"] is False
    assert record["relevant_document_ids"] == []
    assert record["context_document_ids"] == ["rag_evidence"]
    assert record["review"]["insufficient_evidence_handled"] is None


def test_mock_output_is_labeled_and_pending_reviews_do_not_create_quality_scores(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import summarize_reviews

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    assert bundle["run"]["generation"]["mode"] == "mock"
    assert all(record["generation"]["mode"] == "mock" for record in bundle["records"])
    report = summarize_reviews(bundle)
    assert report["reviewed_count"] == 0 and report["pending_count"] == 2
    assert report["human_review_metrics"] is None
    assert report["real_llm_quality_metrics"] is None
    assert "尚无已审核评分" in report["message"]


def complete_review(record):
    review = record["review"]
    review.update(status="completed", reviewer="unit-test-reviewer", reviewed_at="2026-10-20T12:00:00+08:00",
                  faithfulness=1, answer_relevance=0, factual_correctness=2,
                  factual_correctness_basis="单元测试评审夹具，仅验证汇总，不代表真实人工评审。",
                  unsupported_claims=[], review_notes="仅供单元测试。")
    for fact in review["required_facts_covered"]:
        fact.update(covered=False, answer_quote=None)
    if not record["answerable"]:
        review["insufficient_evidence_handled"] = False


def test_review_save_reload_and_summary_keep_dimensions_separate(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results, load_qa_results, summarize_reviews

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    complete_review(bundle["records"][0])
    path = tmp_path / "reviewed.json"
    save_qa_results(bundle, path)
    restored = load_qa_results(path)
    assert restored == bundle
    report = summarize_reviews(restored)
    assert report["reviewed_count"] == 1 and report["pending_count"] == 1
    assert report["human_review_metrics"]["faithfulness_mean"] == 1
    assert report["human_review_metrics"]["answer_relevance_mean"] == 0
    assert report["human_review_metrics"]["factual_correctness_mean"] == 2
    assert report["human_review_metrics"]["required_facts_coverage"] == 0
    assert report["real_llm_quality_metrics"] is None  # 人工审核 Mock 输出也不是实际模型质量。


@pytest.mark.parametrize("bad_case", ["bool_score", "out_of_range", "fake_quote", "missing_reviewer", "missing_basis", "missing_fact", "missing_refusal_review", "unreviewed_claims"])
def test_completed_review_requires_valid_scores_and_traceable_evidence(tmp_path, qa_raw, corpus, bad_case):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    record = bundle["records"][1]
    complete_review(record)
    review = record["review"]
    if bad_case == "bool_score":
        review["faithfulness"] = True
    elif bad_case == "out_of_range":
        review["factual_correctness"] = 3
    elif bad_case == "fake_quote":
        review["unsupported_claims"] = [{"claim": "编造", "answer_quote": "答案中没有这句话", "reason": "无证据", "context_document_ids": []}]
    elif bad_case == "missing_reviewer":
        review["reviewer"] = None
    elif bad_case == "missing_basis":
        review["factual_correctness_basis"] = None
    elif bad_case == "missing_fact":
        review["required_facts_covered"] = []
    elif bad_case == "missing_refusal_review":
        review["insufficient_evidence_handled"] = None
    elif bad_case == "unreviewed_claims":
        review["unsupported_claims"] = None
    with pytest.raises(ValueError):
        save_qa_results(bundle, tmp_path / "bad.json")


def test_review_cannot_silently_change_captured_answer_or_generation_mode(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    for field, value in [("answer", "被修改的答案"), ("generation", {"mode": "real_api"})]:
        changed = deepcopy(bundle)
        changed["records"][0][field] = value
        with pytest.raises(ValueError):
            save_qa_results(changed, tmp_path / "changed.json")


def test_partial_scores_are_not_aggregated(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import summarize_reviews

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    bundle["records"][0]["review"].update(status="in_progress", faithfulness=2)
    assert summarize_reviews(bundle)["human_review_metrics"] is None


def test_unknown_actual_document_id_is_rejected_before_generation(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.ingestion.document import Document

    class Retriever:
        def search(self, query):
            return [SearchResult(Document("unknown", {"eval_id": "unknown"}), 1, 1)]

    class NeverGenerate(BaseLLM):
        def generate(self, prompt):
            pytest.fail("未知来源不应进入生成")

    with pytest.raises(ValueError):
        collect_qa(RAGPipeline(Retriever(), NeverGenerate()), load_fixture(tmp_path, qa_raw, corpus), corpus)


def test_capture_refuses_to_guess_context_when_prompt_cannot_be_verified(tmp_path, qa_raw, corpus, monkeypatch):
    import app.pipeline
    from app.evaluation.qa_evaluator import collect_qa

    monkeypatch.setattr(app.pipeline, "build_prompt", lambda question, context: "无法识别的提示词")
    with pytest.raises(ValueError):
        collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)


def test_empty_retrieval_still_records_actual_empty_context(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa

    class EmptyRetriever:
        def search(self, query):
            return []

    bundle = collect_qa(RAGPipeline(EmptyRetriever(), MockLLM()), load_fixture(tmp_path, qa_raw, corpus), corpus)
    assert bundle["records"][0]["context"] == ""
    assert bundle["records"][0]["prompt"] is None
    assert bundle["records"][0]["llm_called"] is False
    assert bundle["records"][0]["context_document_ids"] == []


@pytest.mark.parametrize("rerank_empty", [False, True])
def test_qa_no_evidence_records_no_model_call_or_prompt_and_roundtrips(tmp_path, qa_raw, corpus, rerank_empty):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import load_qa_results, save_qa_results, summarize_reviews

    class NeverGenerate(BaseLLM):
        def generate(self, prompt):
            pytest.fail("空候选不应调用 LLM")

    class EmptyRetriever:
        def search(self, question):
            return []

    class EmptyReranker:
        accepts_results = True

        def rerank(self, question, candidates, top_k):
            return []

    pipeline = (make_pipeline(corpus, llm=NeverGenerate(), reranker=EmptyReranker()) if rerank_empty
                else RAGPipeline(EmptyRetriever(), NeverGenerate()))
    bundle = collect_qa(pipeline, load_fixture(tmp_path, qa_raw, corpus), corpus)
    for record in bundle["records"]:
        assert record["pipeline_status"] == "no_evidence"
        assert record["llm_called"] is False
        assert record["prompt"] is None and record["context"] == ""
        assert record["context_document_ids"] == []
        assert "没有检索到" in record["answer"]
    path = tmp_path / "no_evidence.json"
    save_qa_results(bundle, path)
    assert load_qa_results(path) == bundle
    assert summarize_reviews(bundle)["real_llm_quality_metrics"] is None


def test_reviewed_no_evidence_is_not_counted_as_real_llm_quality(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import summarize_reviews
    from app.generation.llm import OpenAILLM

    class UnusedRealLLM(OpenAILLM):
        # 不创建 SDK、不读环境变量；只验证已配置真实 Provider 但未调用的情况。
        def __init__(self):
            self.model = "test-model"

        def generate(self, prompt):
            pytest.fail("空候选不得调用已配置的真实 Provider")

    class EmptyRetriever:
        def search(self, question):
            return []

    bundle = collect_qa(RAGPipeline(EmptyRetriever(), UnusedRealLLM()), load_fixture(tmp_path, qa_raw, corpus), corpus)
    complete_review(bundle["records"][0])
    summary = summarize_reviews(bundle)
    assert summary["human_review_metrics"] is not None
    assert summary["real_llm_quality_metrics"] is None


def test_mixed_qa_summary_uses_only_generated_answers_for_real_llm_metrics(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import summarize_reviews, save_qa_results, load_qa_results
    from app.generation.llm import OpenAILLM

    class FakeRealLLM(OpenAILLM):
        def __init__(self):
            self.model = "test-model"
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return "仅供汇总测试的回答"

    llm = FakeRealLLM()
    pipeline = make_pipeline(corpus, llm=llm)
    original_search = pipeline.retriever.search
    pipeline.retriever.search = lambda query: [] if "显卡" in query else original_search(query)
    bundle = collect_qa(pipeline, load_fixture(tmp_path, qa_raw, corpus), corpus)
    assert llm.calls == 1
    assert [record["pipeline_status"] for record in bundle["records"]] == ["answered", "no_evidence"]
    for index, record in enumerate(bundle["records"]):
        complete_review(record)
        record["review"]["faithfulness"] = index * 2
    path = tmp_path / "mixed_review.json"
    save_qa_results(bundle, path)
    summary = summarize_reviews(load_qa_results(path))
    assert summary["human_review_metrics"]["faithfulness_mean"] == 1
    assert summary["real_llm_quality_metrics"]["faithfulness_mean"] == 0
    assert summary["no_evidence_count"] == 1


@pytest.mark.parametrize("bad_case", ["false_call", "unknown_status", "false_no_evidence"])
def test_qa_status_and_model_call_flags_must_match_capture(tmp_path, qa_raw, corpus, bad_case):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import capture_digest, validate_qa_results

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    record = bundle["records"][0]
    if bad_case == "false_call":
        record["llm_called"] = False
    elif bad_case == "unknown_status":
        record["pipeline_status"] = "unknown"
    else:
        record.update(pipeline_status="no_evidence", llm_called=False)
    # 即使重算指纹，字段之间矛盾的记录仍不允许保存。
    bundle["capture_sha256"] = capture_digest(bundle)
    with pytest.raises(ValueError):
        validate_qa_results(bundle)


def test_cli_summary_can_read_pending_reviews_without_models(tmp_path, qa_raw, corpus, capsys, monkeypatch):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results

    path = tmp_path / "qa_records.json"
    save_qa_results(collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus), path)
    script = ROOT / "scripts/evaluate_qa.py"
    main = runpy.run_path(str(script))["main"]
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [str(script), "summarize", "--input", str(path), "--output", str(output)])
    main()
    assert "尚无已审核评分" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["real_llm_quality_metrics"] is None


@pytest.mark.parametrize("relative_path, expected", [
    ("data/evaluation/retrieval_eval.json", "085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71"),
    ("results/day7_retrieval.json", "8c49a887f05ad146311b84bff3dc83fd952baf18c4e9d059b985dfc4d172e2e6"),
    ("results/day7_retrieval.txt", "b3e02ebf24067f329907cc5a4232779f521f72ba74249614a3f4650126569445"),
])
def test_day7_frozen_data_and_reports_remain_byte_identical(relative_path, expected):
    assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_llm_label_and_recording_do_not_expose_private_configuration(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa

    llm = MockLLM()
    llm.model = "real-looking-model-name"
    llm.api_key = "unit-test-secret"
    bundle = collect_qa(make_pipeline(corpus, llm=llm), load_fixture(tmp_path, qa_raw, corpus), corpus)
    assert bundle["run"]["generation"]["mode"] == "mock"
    assert bundle["run"]["generation"]["model"] == "MockLLM"
    assert "unit-test-secret" not in json.dumps(bundle)


def test_save_never_overwrites_existing_captures(tmp_path, qa_raw, corpus):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    path = tmp_path / "records.json"
    save_qa_results(bundle, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_qa_results(bundle, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad_field", ["review_notes", "fact_notes"])
def test_pending_review_notes_still_require_text(tmp_path, qa_raw, corpus, bad_field):
    from app.evaluation.qa_evaluator import collect_qa
    from app.evaluation.qa_review import save_qa_results

    bundle = collect_qa(make_pipeline(corpus), load_fixture(tmp_path, qa_raw, corpus), corpus)
    review = bundle["records"][0]["review"]
    if bad_field == "review_notes":
        review["review_notes"] = {"invalid": "not text"}
    else:
        review["required_facts_covered"][0]["notes"] = []
    with pytest.raises(ValueError):
        save_qa_results(bundle, tmp_path / "bad_notes.json")
