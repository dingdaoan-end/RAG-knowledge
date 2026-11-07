"""学习 Demo：Hybrid Top-4 -> Cross-Encoder -> Top-3；--fake 无需下载重排模型。"""

import argparse

from app.embedding.embedder import TextEmbedder
from app.ingestion.document import Document
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from app.retrieval.retriever import FaissRetriever
from app.retrieval.text_retriever import TextRetriever


DOCUMENTS = [
    Document("RAG、大模型、幻觉、检索增强生成是本文的关键词。本文是这些术语的索引，不解释 RAG 为什么可以降低大模型幻觉。", {"source": "A.txt", "chunk_id": 0}),
    Document("检索增强生成把外部知识库中找到的证据放入提示词，交给语言模型参考。模型据此组织回答，可以减少缺乏事实依据的编造。", {"source": "B.txt", "chunk_id": 0}),
    Document("RAG 通过检索外部知识并提供给大语言模型，减少模型仅依赖参数知识生成答案的情况，从而降低幻觉，但不能保证完全消除幻觉。", {"source": "C.txt", "chunk_id": 0}),
    Document("FAISS 用于高维向量相似度搜索，IndexFlatIP 进行精确内积搜索。", {"source": "D.txt", "chunk_id": 0}),
    Document("CUDA 是 GPU 并行计算平台，需要根据错误日志排查 CUDA 环境问题。", {"source": "E.txt", "chunk_id": 0}),
]
QUERY = "RAG 为什么可以降低大模型幻觉？"
CANDIDATE_K = 4
TOP_K = 3


class DemoScorer:
    """预设候选分数，仅演示排序过程，不用于证明模型相关性效果。"""

    def predict(self, pairs, **kwargs):
        return [0.62, 0.95, 0.71, 0.10][:len(pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="使用明确标注的预设重排分数")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--local-files-only", action="store_true", help="重排模型仅从本地缓存加载")
    args = parser.parse_args()

    print(f"Query: {QUERY}")
    print(f"Corpus={len(DOCUMENTS)}, Hybrid Top-N={CANDIDATE_K}, Rerank Top-K={TOP_K}, alpha=0.7")
    print("Scorer: " + ("FAKE：预设分数，仅验证流程" if args.fake else args.reranker_model))
    for document in DOCUMENTS:
        print(f"{document.metadata['source']}: {document.text}")

    embedder = TextEmbedder()
    vector = TextRetriever(FaissRetriever(DOCUMENTS, embedder.embed_documents(DOCUMENTS)), embedder)
    bm25 = BM25Retriever(DOCUMENTS)
    hybrid = HybridRetriever(vector, bm25, alpha=0.7, top_k=CANDIDATE_K)
    for name, retriever in [("Vector", vector), ("BM25", bm25)]:
        print(f"\n{name} Top-{CANDIDATE_K}: rank | document | raw score")
        for result in retriever.search(QUERY, top_k=CANDIDATE_K):
            print(f"{result.rank} | {result.document.metadata['source']} | {result.score:.6f}")

    candidates = hybrid.search(QUERY)
    print("\nHybrid Top-N: rank | document | hybrid score")
    for result in candidates:
        print(f"{result.rank} | {result.document.metadata['source']} | {result.final_score:.6f}")

    reranker = CrossEncoderReranker(
        model_name=args.reranker_model, local_files_only=args.local_files_only,
        scorer=DemoScorer() if args.fake else None,
    )
    results = reranker.rerank(QUERY, candidates, top_k=TOP_K)
    print(f"\nCross-Encoder input pairs={len(candidates)}（只使用 Hybrid 候选，未扫描全库）")
    print("Rerank Top-K: final rank | document | hybrid score | reranker score | original rank")
    for result in results:
        print(f"{result.rank} | {result.document.metadata['source']} | {result.hybrid_score:.6f} | {result.rerank_score:.6f} | {result.retrieval_rank}")
    print("Hybrid order: " + " > ".join(result.document.metadata["source"] for result in candidates))
    print("Rerank order: " + " > ".join(result.document.metadata["source"] for result in results))
    print("重排分数为 logit（fake 模式除外），不是概率；Hybrid 分数保持不变。")


if __name__ == "__main__":
    main()
