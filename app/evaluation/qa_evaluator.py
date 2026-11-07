"""通过独立包装层运行现有 RAGPipeline，采集实际提示词与上下文。"""

from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from app.generation.context_builder import build_context
from app.generation.llm import BaseLLM, MockLLM, OpenAILLM
from app.generation.prompt import build_prompt
from app.pipeline import RAGPipeline
from app.retrieval.retriever import SearchResult
from .dataset import EvaluationDataset
from .qa_dataset import QADataset
from .qa_review import capture_digest, pending_review
from .retrieval_metrics import validate_identifier


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_info(llm: BaseLLM) -> dict:
    # 模式由实际对象决定，不接受调用方把 Mock 声明成真实模型。
    mode = "mock" if isinstance(llm, MockLLM) else "real_api" if isinstance(llm, OpenAILLM) else "custom_unverified"
    model = "MockLLM" if mode == "mock" else getattr(llm, "model", None)
    return {"mode": mode, "implementation": type(llm).__name__, "model": model if isinstance(model, str) else None}


def _component_info(component) -> dict | None:
    if component is None:
        return None
    info = {"implementation": type(component).__name__}
    for key in ("top_k", "alpha", "model_name", "batch_size", "max_length", "device", "local_files_only"):
        value = getattr(component, key, None)
        if isinstance(value, (str, int, float, bool)):
            info[key] = value
    # 只记录白名单字段，不序列化 SDK client、Settings 或环境变量。
    return info


def _document_info(document, registry: dict) -> dict:
    document_id = document.metadata.get("eval_id")
    validate_identifier(document_id)
    if document_id not in registry:
        raise ValueError(f"无法映射实际上下文文档: {document_id}")
    original = registry[document_id]
    if document.text != original.text or any(document.metadata.get(key) != original.metadata[key] for key in ("source", "chunk_id")):
        raise ValueError(f"文档 ID 的文本或来源不一致: {document_id}")
    return {"id": document_id, "text": document.text, "metadata": deepcopy(document.metadata)}


class _TraceRetriever:
    def __init__(self, retriever, registry):
        self.delegate, self.registry = retriever, registry
        self.results, self.query = [], None

    def search(self, query):
        results = self.delegate.search(query)
        for result in results:
            _document_info(result.document, self.registry)
        self.query, self.results = query, deepcopy(results)
        return results


class _TraceReranker:
    def __init__(self, reranker, registry):
        self.delegate, self.registry = reranker, registry
        self.accepts_results = getattr(reranker, "accepts_results", False)
        self.results = []

    def rerank(self, *args, **kwargs):
        results = self.delegate.rerank(*args, **kwargs)
        for item in results:
            _document_info(item.document if self.accepts_results else item, self.registry)
        self.results = deepcopy(results)
        return results


class _TraceLLM(BaseLLM):
    def __init__(self, llm):
        self.delegate, self.prompt = llm, None

    def generate(self, prompt):
        self.prompt = prompt
        return self.delegate.generate(prompt)


def collect_qa(pipeline: RAGPipeline, dataset: QADataset, corpus: EvaluationDataset) -> dict:
    if dataset.corpus_sha256 != corpus.sha256:
        raise ValueError("QA 数据与语料版本不一致")
    start = perf_counter()
    generation = _generation_info(pipeline.llm)
    run = {
        "id": uuid4().hex, "started_at": _utc_now(),
        "qa_dataset": {"version": dataset.version, "sha256": dataset.sha256, "metadata": dataset.metadata},
        "corpus_sha256": corpus.sha256, "generation": generation,
        "pipeline": {"retriever": _component_info(pipeline.retriever), "reranker": _component_info(pipeline.reranker),
                     "query_rewriter": _component_info(pipeline.query_rewriter), "context_top_k": pipeline.top_k},
        "quality_scope": "mock_workflow_only" if generation["mode"] == "mock" else "pending_human_review",
    }
    # Hybrid -> TextRetriever -> Embedder。只补充可验证的模型配置，不捕获密钥。
    vector = getattr(pipeline.retriever, "vector_retriever", pipeline.retriever)
    embedder = getattr(vector, "embedder", None)
    if embedder is not None:
        run["pipeline"]["embedding"] = _component_info(embedder)
        model = getattr(embedder, "model", None)
        if model is not None:
            run["pipeline"]["embedding"]["model_name"] = model[0].auto_model.config._name_or_path
    registry = {document.metadata["eval_id"]: document for document in corpus.documents}
    records = []
    for sample in dataset.samples:
        sample_start, started_at = perf_counter(), _utc_now()
        retriever = _TraceRetriever(pipeline.retriever, registry)
        reranker = _TraceReranker(pipeline.reranker, registry) if pipeline.reranker else None
        llm = _TraceLLM(pipeline.llm)
        # 新建评估实例复用同一批组件；不修改生产实例，不重新实现检索或生成流程。
        traced = RAGPipeline(retriever, llm, query_rewriter=pipeline.query_rewriter, reranker=reranker, top_k=pipeline.top_k)
        outcome = traced.ask_with_result(sample["question"])
        answer = outcome.answer
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Pipeline 未返回非空答案")
        results = retriever.results
        if reranker:
            if reranker.accepts_results:
                results = reranker.results
            else:
                by_eval_id = {result.document.metadata["eval_id"]: result for result in results}
                results = [SearchResult(document, by_eval_id[document.metadata["eval_id"]].score, rank)
                           for rank, document in enumerate(reranker.results, 1)]
        if pipeline.top_k is not None:
            results = results[:pipeline.top_k]
        context = build_context(results)
        if outcome.status == "no_evidence":
            if results or llm.prompt is not None:
                raise ValueError("证据不足结果不能包含候选或实际 LLM 调用")
            context = ""  # 没有向 LLM 提供上下文，不伪造实际 Prompt。
        elif llm.prompt != build_prompt(sample["question"], context):
            raise ValueError("无法核对实际 LLM Prompt 与上下文，拒绝猜测或保存不可靠来源")
        documents = [{**_document_info(result.document, registry), "rank": result.rank, "score": float(result.score)} for result in results]
        ids = [document["id"] for document in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("实际上下文来源 ID 不能重复")
        records.append({
            **deepcopy(sample), "answer": answer, "prompt": llm.prompt, "context": context,
            "pipeline_status": outcome.status, "llm_called": llm.prompt is not None,
            "context_documents": documents, "context_document_ids": ids, "retrieval_query": retriever.query,
            "generation": deepcopy(generation), "started_at": started_at, "elapsed_seconds": perf_counter() - sample_start,
            "review": pending_review(sample["required_facts"]),
        })
    run.update(finished_at=_utc_now(), elapsed_seconds=perf_counter() - start)
    bundle = {"schema_version": 1, "run": run, "records": records}
    bundle["capture_sha256"] = capture_digest(bundle)
    return bundle
