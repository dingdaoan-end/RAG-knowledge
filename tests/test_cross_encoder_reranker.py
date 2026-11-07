from dataclasses import asdict

import numpy as np
import pytest

from app.ingestion.document import Document
from app.retrieval.hybrid import HybridSearchResult


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


@pytest.fixture
def candidates():
    return [
        HybridSearchResult(Document(text, {"source": f"{text}.txt", "chunk_id": 0}),
                           0.8, 2.5, 0.9, 0.7, score, rank)
        for rank, (text, score) in enumerate([("A", 0.91), ("B", 0.87), ("C", 0.80)], 1)
    ]


def test_cross_encoder_sorts_pairs_and_preserves_retrieval_information(candidates):
    from app.retrieval.reranker import CrossEncoderReranker, RerankResult

    scorer = FakeScorer([0.62, 0.95, 0.71])
    before = [asdict(result) for result in candidates]
    results = CrossEncoderReranker(scorer=scorer, batch_size=2).rerank("问题", candidates)
    assert all(isinstance(result, RerankResult) for result in results)
    assert [result.document.text for result in results] == ["B", "C", "A"]
    assert [result.rerank_score for result in results] == [0.95, 0.71, 0.62]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].document is candidates[1].document
    assert results[0].final_score == results[0].hybrid_score == 0.87
    assert results[0].vector_score == 0.8
    assert results[0].bm25_score == 2.5
    assert results[0].normalized_vector_score == 0.9
    assert results[0].normalized_bm25_score == 0.7
    assert results[0].retrieval_rank == 2
    assert results[0].score == 0.95
    assert [asdict(result) for result in candidates] == before
    assert scorer.calls[0][0] == [("问题", "A"), ("问题", "B"), ("问题", "C")]
    assert scorer.calls[0][1]["batch_size"] == 2


@pytest.mark.parametrize("top_k, expected", [(1, ["B"]), (2, ["B", "C"]), (3, ["B", "C", "A"]), (10, ["B", "C", "A"])])
def test_top_k_is_applied_after_scoring_all_candidates(candidates, top_k, expected):
    from app.retrieval.reranker import CrossEncoderReranker

    scorer = FakeScorer([0.62, 0.95, 0.71])
    results = CrossEncoderReranker(scorer=scorer).rerank("问题", candidates, top_k=top_k)
    assert [result.document.text for result in results] == expected
    assert len(scorer.calls[0][0]) == 3


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3", None])
def test_invalid_top_k_is_rejected_before_scoring(candidates, top_k):
    from app.retrieval.reranker import CrossEncoderReranker

    scorer = FakeScorer([1, 2, 3])
    with pytest.raises(ValueError, match="top_k"):
        CrossEncoderReranker(scorer=scorer).rerank("问题", candidates, top_k=top_k)
    assert scorer.calls == []


@pytest.mark.parametrize("query", ["", " \n", None])
def test_empty_query_is_rejected(candidates, query):
    from app.retrieval.reranker import CrossEncoderReranker

    with pytest.raises(ValueError, match="query"):
        CrossEncoderReranker(scorer=FakeScorer([1, 2, 3])).rerank(query, candidates)


def test_empty_candidates_do_not_load_or_call_a_model():
    from app.retrieval.reranker import CrossEncoderReranker

    scorer = FakeScorer([])
    assert CrossEncoderReranker(scorer=scorer).rerank("问题", []) == []
    assert scorer.calls == []
    assert CrossEncoderReranker(model_name="not-downloaded").rerank("问题", []) == []


def test_single_candidate_accepts_a_negative_logit(candidates):
    from app.retrieval.reranker import CrossEncoderReranker

    result = CrossEncoderReranker(scorer=FakeScorer([-3.5])).rerank("问题", candidates[:1], top_k=10)[0]
    assert result.document is candidates[0].document
    assert result.rerank_score == -3.5
    assert result.rank == 1


def test_duplicate_document_is_rejected_before_scoring(candidates):
    from app.retrieval.reranker import CrossEncoderReranker

    scorer = FakeScorer([1, 2])
    with pytest.raises(ValueError, match="重复"):
        CrossEncoderReranker(scorer=scorer).rerank("问题", [candidates[0], candidates[0]])
    assert scorer.calls == []


def test_same_text_in_distinct_document_objects_is_not_deduplicated(candidates):
    from dataclasses import replace
    from app.retrieval.reranker import CrossEncoderReranker

    copy = replace(candidates[0], document=Document("A", {"source": "other.txt", "chunk_id": 0}))
    results = CrossEncoderReranker(scorer=FakeScorer([0.2, 0.8])).rerank("问题", [candidates[0], copy])
    assert len(results) == 2
    assert results[0].document is copy.document


@pytest.mark.parametrize("scores", [[np.nan, 1, 2], [np.inf, 1, 2], [-np.inf, 1, 2], [1, 2], [[1, 2, 3]], [[1, 2], [3, 4], [5, 6]], ["invalid", 1, 2], [1j, 1, 2]])
def test_invalid_model_scores_raise_value_error(candidates, scores):
    from app.retrieval.reranker import CrossEncoderReranker

    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=FakeScorer(scores)).rerank("问题", candidates)


def test_equal_scores_keep_candidate_order(candidates):
    from app.retrieval.reranker import CrossEncoderReranker

    results = CrossEncoderReranker(scorer=FakeScorer([0.5, 0.5, 0.5])).rerank("问题", candidates)
    assert [result.document.text for result in results] == ["A", "B", "C"]


def test_model_configuration_is_used_and_model_is_loaded_once(monkeypatch, candidates):
    import sentence_transformers
    from torch.nn import Identity
    from app.retrieval.reranker import CrossEncoderReranker

    scorer = FakeScorer([1, 2, 3])
    loads = []

    def load_model(name, **kwargs):
        loads.append((name, kwargs))
        return scorer

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", load_model)
    reranker = CrossEncoderReranker(model_name="custom/model", max_length=256, device="cpu", local_files_only=True)
    assert loads == []
    reranker.rerank("问题", candidates)
    reranker.rerank("另一问题", candidates)
    assert len(loads) == 1
    assert loads[0][0] == "custom/model"
    assert loads[0][1]["max_length"] == 256
    assert loads[0][1]["device"] == "cpu"
    assert loads[0][1]["local_files_only"] is True
    assert isinstance(loads[0][1]["default_activation_function"], Identity)


@pytest.mark.parametrize("setting", [{"batch_size": 0}, {"batch_size": True}, {"max_length": 0}, {"model_name": ""}])
def test_invalid_configuration_is_rejected(setting):
    from app.retrieval.reranker import CrossEncoderReranker

    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=FakeScorer([]), **setting)


@pytest.mark.parametrize("streaming", [False, True])
def test_pipeline_reranks_only_hybrid_top_n_then_builds_context(streaming):
    from app.generation.llm import BaseLLM
    from app.pipeline import RAGPipeline
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever
    from app.retrieval.retriever import FaissRetriever
    from app.retrieval.text_retriever import TextRetriever
    from app.retrieval.reranker import CrossEncoderReranker

    documents = [Document(text, {"source": f"{index}.txt"}) for index, text in enumerate(["RAG 检索", "RAG 生成", "RAG 向量", "足球"])]

    class Embedder:
        def embed_query(self, query):
            return np.array([[1, 0]], dtype="float32")

    vector = TextRetriever(FaissRetriever(documents, np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype="float32")), Embedder())
    hybrid = HybridRetriever(vector, BM25Retriever(documents), top_k=2)
    candidates = hybrid.search("RAG")
    scorer = FakeScorer([0.1, 0.9])

    class RecordingLLM(BaseLLM):
        def generate(self, prompt):
            self.prompt = prompt
            return "答案"

        def stream_generate(self, prompt):
            yield self.generate(prompt)

    llm = RecordingLLM()
    pipeline = RAGPipeline(hybrid, llm, reranker=CrossEncoderReranker(scorer=scorer), top_k=1)
    answer = "".join(pipeline.stream_ask("RAG")) if streaming else pipeline.ask("RAG")
    assert answer == "答案"
    assert scorer.calls[0][0] == [("RAG", result.document.text) for result in candidates]
    assert len(scorer.calls[0][0]) == 2  # 语料有 4 篇，但只评分 Hybrid Top-2。
    assert f"source: {candidates[1].document.metadata['source']}" in llm.prompt
    assert f"source: {candidates[0].document.metadata['source']}" not in llm.prompt
    assert "rank: 1" in llm.prompt and "score: 0.9" in llm.prompt


def test_column_scores_are_accepted(candidates):
    from app.retrieval.reranker import CrossEncoderReranker

    results = CrossEncoderReranker(scorer=FakeScorer([[0.62], [0.95], [0.71]])).rerank("问题", candidates)
    assert [result.document.text for result in results] == ["B", "C", "A"]


def test_model_loading_failure_is_not_silently_replaced(monkeypatch, candidates):
    import sentence_transformers
    from app.retrieval.reranker import CrossEncoderReranker

    def unavailable(*args, **kwargs):
        raise OSError("model unavailable")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", unavailable)
    with pytest.raises(OSError, match="model unavailable"):
        CrossEncoderReranker().rerank("问题", candidates)


def test_cli_can_select_cross_encoder_and_configure_model(monkeypatch, tmp_path, capsys):
    import main
    from app.retrieval.reranker import CrossEncoderReranker

    path = tmp_path / "knowledge.txt"
    path.write_text("RAG 是检索增强生成。", encoding="utf-8")
    selected_models = []
    scorer = FakeScorer([0.9])

    def make_reranker(model_name):
        selected_models.append(model_name)
        return CrossEncoderReranker(model_name=model_name, scorer=scorer)

    monkeypatch.setattr(main, "CrossEncoderReranker", make_reranker, raising=False)
    monkeypatch.setattr("sys.argv", ["main.py", "--file", str(path), "--mock", "--retrieval", "bm25",
                                    "--reranker", "cross-encoder", "--reranker-model", "custom/model", "--query", "RAG"])
    main.main()
    assert selected_models == ["custom/model"]
    assert scorer.calls[0][0] == [("RAG", "RAG 是检索增强生成。")]
    assert "这是基于知识库资料生成的测试回答" in capsys.readouterr().out


def test_pipeline_keeps_legacy_rerankers_without_base_class_compatible(candidates):
    from app.generation.llm import BaseLLM
    from app.pipeline import RAGPipeline

    class Retriever:
        def search(self, query):
            return candidates

    class LegacyReranker:
        def rerank(self, query, documents):
            assert all(isinstance(document, Document) for document in documents)
            return list(reversed(documents))

    class PromptLLM(BaseLLM):
        def generate(self, prompt):
            return prompt

    prompt = RAGPipeline(Retriever(), PromptLLM(), reranker=LegacyReranker(), top_k=1).ask("问题")
    assert "source: C.txt" in prompt and "source: A.txt" not in prompt
    assert "rank: 1" in prompt and "score: 0.8" in prompt
