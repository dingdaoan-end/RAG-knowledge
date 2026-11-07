import numpy as np
import pytest

from app.ingestion.document import Document
from app.retrieval.retriever import FaissRetriever, SearchResult


@pytest.mark.parametrize("as_dict", [False, True])
def test_context_preserves_metadata_rank_score_and_multiple_results(as_dict):
    from app.generation.context_builder import build_context

    document = Document("RAG 先检索资料，再生成答案。", {"source": "rag.txt", "chunk_id": 0, "category": "学习", "details": {"page": 2}})
    results = [SearchResult(document, 0.85, 2), SearchResult(Document("FAISS 用于向量搜索。", {"source": "faiss.md", "chunk_id": 3}), 0.75, 5)]
    if as_dict:
        results = [{"document": result.document, "score": result.score, "rank": result.rank} for result in results]
    context = build_context(results)
    for expected in ["[Document 2]", "source: rag.txt", "chunk_id: 0", "rank: 2", "score: 0.85", "category: 学习", "'page': 2", "content:\nRAG 先检索资料，再生成答案。", "[Document 5]", "source: faiss.md", "score: 0.75"]:
        assert expected in context
    assert document.metadata == {"source": "rag.txt", "chunk_id": 0, "category": "学习", "details": {"page": 2}}


def test_empty_context_is_explicit():
    from app.generation.context_builder import build_context

    assert build_context([]) == "没有检索到可用的知识库资料。"


def test_prompt_contains_question_context_and_answer_constraints():
    from app.generation.prompt import build_prompt

    prompt = build_prompt("RAG 是什么？", "RAG 是检索增强生成。")
    for expected in ["System:", "Context:\n\nRAG 是检索增强生成。", "Question:\n\nRAG 是什么？", "根据提供的资料回答问题", "不要使用资料之外的信息猜测", "如果资料中没有答案，明确告诉用户"]:
        assert expected in prompt


def test_base_llm_requires_implementation():
    from app.generation.llm import BaseLLM

    with pytest.raises(NotImplementedError):
        BaseLLM().generate("prompt")


def test_mock_llm_returns_test_answer():
    from app.generation.llm import MockLLM

    assert MockLLM().generate("prompt") == "这是基于知识库资料生成的测试回答"


class FixedEmbedder:
    """测试中用已知向量替代耗时的模型推理；FAISS 仍真实运行。"""

    def __init__(self):
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return np.array([[1, 0]], dtype="float32")


def test_text_retriever_encodes_question_and_applies_top_k():
    from app.retrieval.text_retriever import TextRetriever

    documents = [Document("RAG 检索资料", {"source": "rag.txt"}), Document("烹饪", {})]
    embedder = FixedEmbedder()
    retriever = TextRetriever(FaissRetriever(documents, np.eye(2, dtype="float32")), embedder, top_k=1)
    results = retriever.search("RAG 是什么？")
    assert embedder.questions == ["RAG 是什么？"]
    assert len(results) == 1
    assert results[0].document.metadata["source"] == "rag.txt"


def test_pipeline_runs_with_real_faiss_and_mock_llm():
    from app.generation.llm import MockLLM
    from app.pipeline import RAGPipeline
    from app.retrieval.text_retriever import TextRetriever

    documents = [Document("RAG 先检索，再生成。", {"source": "rag.txt", "chunk_id": 0})]
    retriever = TextRetriever(FaissRetriever(documents, np.array([[1, 0]], dtype="float32")), FixedEmbedder())
    assert RAGPipeline(retriever, MockLLM()).ask("RAG 是什么？") == "这是基于知识库资料生成的测试回答"


def test_pipeline_passes_retrieved_evidence_and_question_to_llm():
    from app.generation.llm import BaseLLM
    from app.pipeline import RAGPipeline

    class MemoryRetriever:
        def search(self, question):
            assert question == "RAG 是什么？"
            return [SearchResult(Document("RAG 是检索增强生成。", {"source": "notes.md", "chunk_id": 4}), 0.9, 1)]

    class RecordingLLM(BaseLLM):
        def generate(self, prompt):
            self.prompt = prompt
            return "根据资料，RAG 是检索增强生成。"

    llm = RecordingLLM()
    assert RAGPipeline(MemoryRetriever(), llm).ask("RAG 是什么？") == "根据资料，RAG 是检索增强生成。"
    for expected in ["Question:\n\nRAG 是什么？", "source: notes.md", "chunk_id: 4", "score: 0.9", "RAG 是检索增强生成。"]:
        assert expected in llm.prompt


def test_pipeline_returns_no_evidence_without_calling_llm():
    from app.generation.llm import BaseLLM
    from app.pipeline import RAGPipeline

    class EmptyRetriever:
        def search(self, question):
            return []

    class RecordingLLM(BaseLLM):
        def generate(self, prompt):
            pytest.fail("空候选不得进入生成流程")

    answer = RAGPipeline(EmptyRetriever(), RecordingLLM()).ask("未知问题")
    assert isinstance(answer, str) and "没有检索到" in answer
