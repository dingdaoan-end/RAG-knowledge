"""采集现有 Pipeline 的问答与证据，或汇总已完成的人工评审。"""

import argparse
import json
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_dataset
from app.evaluation.qa_dataset import load_qa_dataset
from app.evaluation.qa_review import load_qa_results, save_qa_results, summarize_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="采集实际回答，不打分")
    collect.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/qa_eval.json")
    collect.add_argument("--corpus", type=Path, default=ROOT / "data/evaluation/retrieval_eval.json")
    collect.add_argument("--output", type=Path, default=ROOT / "results/day8_qa_mock.json")
    collect.add_argument("--llm", choices=["mock", "real"], default="mock")
    collect.add_argument("--candidate-k", type=int, default=10)
    collect.add_argument("--top-k", type=int, default=3)
    collect.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    collect.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    collect.add_argument("--local-files-only", action="store_true")
    summary = commands.add_parser("summarize", help="只汇总已完成人工评审的记录，不调用模型")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, default=ROOT / "results/day8_qa_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出文件已存在，请选择新文件名；不会覆盖历史结果或冻结基线")
    if args.command == "summarize":
        report = summarize_reviews(load_qa_results(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(report["message"])
        print(f"已审核: {report['reviewed_count']}，待审核: {report['pending_count']}")
        print(f"Summary saved: {args.output.resolve()}")
        return

    if args.top_k <= 0 or args.candidate_k < args.top_k:
        parser.error("top-k 必须为正整数，且 candidate-k >= top-k")
    corpus = load_dataset(args.corpus)
    dataset = load_qa_dataset(args.dataset, corpus)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from app.evaluation.qa_evaluator import collect_qa
    from app.embedding.embedder import TextEmbedder
    from app.generation.llm import MockLLM, OpenAILLM
    from app.pipeline import RAGPipeline
    from app.retrieval.hybrid import BM25Retriever, HybridRetriever
    from app.retrieval.retriever import FaissRetriever
    from app.retrieval.text_retriever import TextRetriever
    from app.retrieval.reranker import CrossEncoderReranker

    # 模式需要显式选择；真实 LLM 不可用时正常报错，不静默替换。
    llm = MockLLM() if args.llm == "mock" else OpenAILLM()
    try:
        embedder = TextEmbedder(model_name=args.embedding_model)
        vector = TextRetriever(FaissRetriever(corpus.documents, embedder.embed_documents(corpus.documents)), embedder)
        hybrid = HybridRetriever(vector, BM25Retriever(corpus.documents), alpha=0.7, top_k=args.candidate_k)
        reranker = CrossEncoderReranker(model_name=args.reranker_model, local_files_only=args.local_files_only)
        pipeline = RAGPipeline(hybrid, llm, reranker=reranker, top_k=args.top_k)
        bundle = collect_qa(pipeline, dataset, corpus)
        save_qa_results(bundle, args.output)
    finally:
        llm.close()
    print(f"Generation: {bundle['run']['generation']}")
    for record in bundle["records"]:
        print(f"\n{record['id']}: {record['question']}")
        print(f"answerable={record['answerable']}, context_document_ids={record['context_document_ids']}")
        print(f"answer: {record['answer']}")
        print("人工评审: pending，所有评分尚未填写")
    print(f"\nRecords saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
