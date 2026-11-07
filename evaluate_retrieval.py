import argparse
import json
from pathlib import Path

import numpy as np

from app.ingestion.document import Document
from app.retrieval.retriever import FaissRetriever
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.retrieval.text_retriever import TextRetriever


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """一个查询的 Recall@K；集合去重，避免重复结果被重复计分。"""
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")
    if not relevant_ids:
        raise ValueError("每个查询至少需要一个相关文档标注")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def evaluate_retrieval(
    retriever,
    query_embeddings: np.ndarray | list[str],
    relevant_ids: list[set[str]],
) -> dict[int, float]:
    """先逐查询计算 Recall，再取算术平均（macro average）。"""
    if not relevant_ids or len(query_embeddings) != len(relevant_ids):
        raise ValueError("查询不能为空，且查询数量必须与标注数量一致")
    totals = {1: 0.0, 3: 0.0, 5: 0.0}
    for query_vector, gold_ids in zip(query_embeddings, relevant_ids):
        # 每个查询只检索一次 Top-5，前缀分别用于 Recall@1/3/5。
        results = retriever.search(query_vector, top_k=5)
        retrieved_ids = [result.document.metadata["id"] for result in results]
        for k in totals:
            totals[k] += recall_at_k(retrieved_ids, gold_ids, k)
    return {k: total / len(relevant_ids) for k, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="计算检索 Recall@1、Recall@3、Recall@5")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data" / "retrieval_eval.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"], default="vector")
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("alpha 必须在 [0, 1] 内")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    # 评估集中的每条 document 已是一个检索单元，无需再次切分。
    documents = [
        Document(item["text"], {"id": item["id"], "source": str(args.dataset.resolve())})
        for item in dataset["documents"]
    ]
    document_ids = {document.metadata["id"] for document in documents}
    if len(document_ids) != len(documents):
        raise ValueError("评估文档 id 必须唯一")
    questions = [item["query"] for item in dataset["queries"]]
    labels = [set(item["relevant_ids"]) for item in dataset["queries"]]
    if not questions or any(not ids or not ids <= document_ids for ids in labels):
        raise ValueError("查询不能为空，相关文档 id 必须存在且标注不能为空")

    if args.retrieval == "bm25":
        retriever = BM25Retriever(documents)
    else:
        from app.embedding.embedder import TextEmbedder

        embedder = TextEmbedder(batch_size=args.batch_size)
        vector = FaissRetriever(documents, embedder.embed_documents(documents))
        retriever = TextRetriever(vector, embedder)
        if args.retrieval == "hybrid":
            retriever = HybridRetriever(retriever, BM25Retriever(documents), alpha=args.alpha)
    metrics = evaluate_retrieval(retriever, questions, labels)
    print(f"Retrieval: {args.retrieval}")
    print(f"Documents: {len(documents)}, Queries: {len(questions)}")
    for k, score in metrics.items():
        print(f"Recall@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
