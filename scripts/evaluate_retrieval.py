"""四方案检索评估入口；与根目录旧版 Recall 脚本独立。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_dataset
from app.evaluation.evaluator import evaluate_methods, format_report, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/retrieval_eval.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/day7_retrieval.json")
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.candidate_k <= 0 or not args.ks or any(k <= 0 for k in args.ks) or args.candidate_k < max(args.ks) or len(set(args.ks)) != len(args.ks):
        parser.error("候选数和 K 必须为正整数，K 不重复，且 candidate-k >= max(ks)")
    if args.output.resolve() == args.dataset.resolve():
        parser.error("output 不能覆盖 Ground Truth 数据集")
    if args.output.exists():
        parser.error(f"输出文件已存在：{args.output}，请指定新的输出路径。不会覆盖历史报告。")
    dataset = load_dataset(args.dataset)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 参数及数据集先验证，再加载真实模型；单元测试不执行此模型创建路径。
    from app.embedding.embedder import TextEmbedder
    from app.retrieval.retriever import FaissRetriever
    from app.retrieval.text_retriever import TextRetriever
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever
    from app.retrieval.reranker import CrossEncoderReranker

    embedder = TextEmbedder(model_name=args.embedding_model, batch_size=args.batch_size)
    vector = TextRetriever(FaissRetriever(dataset.documents, embedder.embed_documents(dataset.documents)), embedder)
    bm25 = BM25Retriever(dataset.documents)
    hybrid = HybridRetriever(vector, bm25, alpha=0.7)
    reranker = CrossEncoderReranker(model_name=args.reranker_model, batch_size=args.reranker_batch_size, local_files_only=args.local_files_only)
    report = evaluate_methods(dataset, vector, bm25, hybrid, reranker, ks=args.ks, candidate_k=args.candidate_k)
    report["run"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_path": str(args.dataset.resolve()),
        "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
        "alpha": 0.7, "embedding_batch_size": args.batch_size, "reranker_batch_size": args.reranker_batch_size,
        "reranker_max_length": reranker.max_length, "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "faiss-cpu", "sentence-transformers", "rank-bm25", "jieba", "torch")},
    }
    save_report(report, args.output)
    print(format_report(report))
    print(f"\nJSON saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
