"""统一比较现有四种方案；不改写查询、不调整检索结果或标注。"""

import json
import os
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile

from .dataset import EvaluationDataset
from .retrieval_metrics import first_relevant_rank, ndcg_at_k, recall_at_k, reciprocal_rank, validate_identifier


METHODS = ("vector", "bm25", "hybrid", "hybrid_reranker")
SCORE_FIELDS = ("vector_score", "bm25_score", "normalized_vector_score", "normalized_bm25_score", "final_score", "rerank_score", "retrieval_score", "retrieval_rank")


def _ranked_rows(results: list, registry: dict, relevance: dict[str, int], candidate_k: int) -> list[dict]:
    if len(results) > candidate_k:
        raise ValueError("检索结果超出约定候选预算")
    rows, seen = [], set()
    for rank, result in enumerate(results, 1):
        document = result.document
        document_id = document.metadata.get("eval_id")
        validate_identifier(document_id)
        if document_id not in registry or document_id in seen:
            raise ValueError(f"返回了未知或重复评估标识: {document_id}")
        original = registry[document_id]
        if document.text != original.text or any(document.metadata.get(key) != original.metadata[key] for key in ("source", "chunk_id")):
            raise ValueError(f"评估 ID 对应的文本或来源不一致: {document_id}")
        seen.add(document_id)
        row = {"id": document_id, "rank": rank, "relevance": relevance.get(document_id, 0), "score": float(result.score)}
        row.update({key: getattr(result, key) for key in SCORE_FIELDS if hasattr(result, key)})
        rows.append(row)
    return rows


def evaluate_methods(
    dataset: EvaluationDataset, vector_retriever, bm25_retriever, hybrid_retriever, reranker,
    ks: list[int] | tuple[int, ...] = (3, 5, 10), candidate_k: int = 10,
) -> dict:
    if type(candidate_k) is not int or candidate_k <= 0:
        raise ValueError("candidate_k 必须是正整数")
    if not ks or any(type(k) is not int or k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks 必须是非空且不重复的正整数列表")
    ks = sorted(ks)
    if candidate_k < max(ks):
        raise ValueError("candidate_k 不能小于要报告的最大 K")
    registry = {document.metadata["eval_id"]: document for document in dataset.documents}
    mrr_name = f"MRR@{candidate_k}"
    report = {
        "schema_version": 1,
        "dataset": {"sha256": dataset.sha256, "documents": len(dataset.documents), "queries": len(dataset.queries), "metadata": dataset.metadata},
        "config": {"ks": ks, "candidate_k": candidate_k, "mrr_name": mrr_name, "relevant_threshold": 1,
                   "unlisted_relevance": 0, "aggregation": "macro_mean", "rank_starts_at": 1,
                   "no_positive_ground_truth": "error", "rerank_complete_hybrid_candidates": True},
        "summary": {}, "queries": [],
    }
    for sample in dataset.queries:
        query = sample["query"]
        relevance = {label["id"]: label["relevance"] for label in sample["relevant_documents"]}
        # 每种方法按相同预算请求；BM25 的少量结果不补齐。
        results = {
            "vector": vector_retriever.search(query, top_k=candidate_k),
            "bm25": bm25_retriever.search(query, top_k=candidate_k),
            "hybrid": hybrid_retriever.search(query, top_k=candidate_k),
        }
        hybrid_candidates = results["hybrid"]
        hybrid_rows = _ranked_rows(hybrid_candidates, registry, relevance, candidate_k)
        results["hybrid_reranker"] = reranker.rerank(query, hybrid_candidates, top_k=len(hybrid_candidates)) if hybrid_candidates else []
        query_report = {"id": sample["id"], "query": query, "relevant_documents": sample["relevant_documents"], "methods": {}}
        for method in METHODS:
            rows = _ranked_rows(results[method], registry, relevance, candidate_k)
            if method == "hybrid_reranker" and {row["id"] for row in rows} != {row["id"] for row in hybrid_rows}:
                raise ValueError("Reranker 必须返回全部且仅来自 Hybrid 的候选")
            ids = [row["id"] for row in rows]
            rr = reciprocal_rank(ids, relevance)
            metrics = {f"Recall@{k}": recall_at_k(ids, relevance, k) for k in ks}
            metrics.update({f"nDCG@{k}": ndcg_at_k(ids, relevance, k) for k in ks})
            query_report["methods"][method] = {
                "candidate_count": len(rows), "ranking": rows, "first_relevant_rank": first_relevant_rank(ids, relevance),
                "rr": rr, "rr_scope": f"RR@{candidate_k}", "metrics": metrics,
            }
        report["queries"].append(query_report)

    for method in METHODS:
        entries = [sample["methods"][method] for sample in report["queries"]]
        report["summary"][method] = {key: fmean(entry["metrics"][key] for entry in entries) for key in entries[0]["metrics"]}
        report["summary"][method][mrr_name] = fmean(entry["rr"] for entry in entries)
        report["summary"][method]["mean_candidate_count"] = fmean(entry["candidate_count"] for entry in entries)
    best = {}
    for key in [f"Recall@{k}" for k in ks] + [mrr_name]:
        maximum = max(report["summary"][method][key] for method in METHODS)
        best[key] = [method for method in METHODS if report["summary"][method][key] == maximum]
    report["analysis"] = {
        "best_methods": best,
        "reranker_ndcg_delta": {f"nDCG@{k}": report["summary"]["hybrid_reranker"][f"nDCG@{k}"] - report["summary"]["hybrid"][f"nDCG@{k}"] for k in ks},
        "reranker_changed_queries": [sample["id"] for sample in report["queries"] if [row["id"] for row in sample["methods"]["hybrid"]["ranking"]] != [row["id"] for row in sample["methods"]["hybrid_reranker"]["ranking"]]],
        "no_relevant_hits": {method: [sample["id"] for sample in report["queries"] if sample["methods"][method]["first_relevant_rank"] is None] for method in METHODS},
    }
    return report


def save_report(report: dict, path: str | Path) -> None:
    """先完成临时文件，再原子发布；任何已有目标均拒绝覆盖。"""
    path = Path(path)
    exists_message = f"输出文件已存在：{path}，请指定新的输出路径。"
    if path.exists():
        raise FileExistsError(exists_message)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # 同目录硬链接只发布已完成内容；目标即使刚被其他进程创建，也不会覆盖。
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(exists_message) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def format_report(report: dict) -> str:
    ks, mrr_name = report["config"]["ks"], report["config"]["mrr_name"]
    keys = [f"Recall@{k}" for k in ks] + [mrr_name] + [f"nDCG@{k}" for k in ks]
    lines = [f"Documents={report['dataset']['documents']}, Queries={report['dataset']['queries']}, candidate_k={report['config']['candidate_k']}",
             "Method | " + " | ".join(keys) + " | Mean candidates"]
    for method, metrics in report["summary"].items():
        lines.append(method + " | " + " | ".join(f"{metrics[key]:.6f}" for key in keys) + f" | {metrics['mean_candidate_count']:.2f}")
    for sample in report["queries"]:
        lines.extend(["", f"{sample['id']}: {sample['query']}", f"Ground truth: {sample['relevant_documents']}"])
        for method, entry in sample["methods"].items():
            ranking = ", ".join(f"{row['rank']}:{row['id']}(rel={row['relevance']})" for row in entry["ranking"])
            metrics = ", ".join(f"{key}={value:.6f}" for key, value in entry["metrics"].items())
            lines.append(f"{method} [{entry['candidate_count']}]: {ranking or '(empty)'}")
            lines.append(f"  first_relevant_rank={entry['first_relevant_rank']}, {entry['rr_scope']}={entry['rr']:.6f}, {metrics}")
    lines.extend(["", "实验摘要（仅适用于本小型数据集）:"])
    for key, methods in report["analysis"]["best_methods"].items():
        lines.append(f"{key} 最高（含并列）: {', '.join(methods)}")
    lines.append(f"Reranker nDCG 变化: {report['analysis']['reranker_ndcg_delta']}")
    lines.append(f"重排顺序变化的 Query: {report['analysis']['reranker_changed_queries']}")
    lines.append(f"各方案完全未命中的 Query: {report['analysis']['no_relevant_hits']}")
    return "\n".join(lines)
