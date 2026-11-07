from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from hashlib import sha256
from ntpath import basename
from typing import Literal

from app.generation.context_builder import build_context
from app.generation.llm import BaseLLM
from app.generation.prompt import build_prompt
from app.query.rewrite import QueryRewriter
from app.retrieval.reranker import Reranker
from app.observability import stage


NO_EVIDENCE_ANSWER = "当前知识库没有检索到可用于回答该问题的资料，因此暂时无法根据现有证据回答。你可以补充相关文档或提供更多问题背景。"


@dataclass(frozen=True)
class RAGResult:
    answer: str
    status: Literal["answered", "no_evidence"]
    sources: list[dict] = field(default_factory=list)


def _source_info(result) -> dict:
    """从最终候选提取展示来源，不改写送入 Context Builder 的元数据。"""
    document = result.document
    metadata = document.metadata
    source = str(metadata.get("source") or "未知来源")
    document_id = metadata.get("document_id") or metadata.get("eval_id")
    if not document_id:
        # 旧文档没有全局 ID；同一 source 的不同 chunk 使用同一兼容 ID。
        document_id = "source_" + sha256(source.encode("utf-8")).hexdigest()
    info = {
        "document_id": str(document_id),
        # ntpath 同时识别 Windows 和 POSIX 分隔符，避免在其他系统泄露路径。
        "source": basename(source.rstrip("/\\")) or "未知来源",
        "chunk_id": metadata.get("chunk_id"),
        "content": document.text,
        "rank": result.rank,
        "score": float(result.score),
    }
    if "page" in metadata:
        info["page"] = metadata["page"]
    return info


class RAGPipeline:
    """编排流程；retriever 只需提供接收问题文本的 search(question)。"""

    def __init__(
        self, retriever, llm: BaseLLM,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None, top_k: int | None = None,
    ) -> None:
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k 必须是正整数")
        self.retriever = retriever
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.top_k = top_k

    def _prepare_prompt(self, question: str) -> tuple[str | None, list[dict]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空")
        query = self.query_rewriter.rewrite(question) if self.query_rewriter else question
        with stage("retrieval"):
            results = self.retriever.search(query)
        if not results:
            return None, []
        if self.reranker:
            with stage("reranker"):
                by_id = {id(result.document): result for result in results}
                accepts_results = getattr(self.reranker, "accepts_results", False)
                if accepts_results:
                    results = self.reranker.rerank(
                        question, results, top_k=self.top_k or len(results) or 1,
                    )
                    documents = [result.document for result in results]
                else:
                    # 兼容既有 Mock 和返回 Document 的旧实现。
                    documents = self.reranker.rerank(question, [result.document for result in results])
                ids = [id(document) for document in documents]
                if len(set(ids)) != len(ids) or any(document_id not in by_id for document_id in ids):
                    raise ValueError("Reranker 只能返回不重复的候选 Document 对象")
                if not accepts_results:
                    # 更新旧接口的 rank，保留原检索 score，不修改原结果。
                    results = [replace(by_id[document_id], rank=rank) for rank, document_id in enumerate(ids, 1)]
        if self.top_k is not None:
            results = results[:self.top_k]
        if not results:
            return None, []
        with stage("context"):
            context = build_context(results)
            # 回答和重排都面向用户原问题；改写结果只用于检索。
            return build_prompt(question, context), [_source_info(result) for result in results]

    def _build_prompt(self, question: str) -> str | None:
        """兼容只需要 Prompt 的既有调用；每次只执行一遍准备流程。"""
        prompt, _ = self._prepare_prompt(question)
        return prompt

    def ask(self, question: str) -> str:
        return self.ask_with_result(question).answer

    def ask_with_result(self, question: str) -> RAGResult:
        with stage("pipeline") as execution:
            prompt, sources = self._prepare_prompt(question)
            if prompt is None:
                execution.status = "no_evidence"
                return RAGResult(NO_EVIDENCE_ANSWER, "no_evidence")
            with stage("llm"):
                answer = self.llm.generate(prompt)
            return RAGResult(answer, "answered", sources)

    def stream_ask(self, question: str) -> Iterator[str]:
        results = self.stream_ask_with_result(question)
        try:
            for result in results:
                yield result.answer
        finally:
            results.close()

    def stream_ask_with_result(self, question: str) -> Iterator[RAGResult]:
        """逐段返回文本和状态；仍延迟执行，关闭时释放上游流。"""
        with stage("pipeline") as execution:
            prompt, sources = self._prepare_prompt(question)
            if prompt is None:
                execution.status = "no_evidence"
                yield RAGResult(NO_EVIDENCE_ANSWER, "no_evidence")
                return
            with stage("llm"):
                tokens = self.llm.stream_generate(prompt)
                try:
                    for token in tokens:
                        yield RAGResult(token, "answered", sources)
                finally:
                    close = getattr(tokens, "close", None)
                    if close is not None:
                        close()
