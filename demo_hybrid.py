"""Day 6 实验：使用现有检索器观察真实分数，不修改检索算法。"""

from app.embedding.embedder import TextEmbedder
from app.ingestion.document import Document
from app.retrieval.hybrid import BM25Retriever, HybridRetriever, tokenize
from app.retrieval.retriever import FaissRetriever
from app.retrieval.text_retriever import TextRetriever


DOCUMENTS = [
    Document("RAG 是一种检索增强生成技术，通过从外部知识库检索相关信息，再将检索结果提供给大语言模型。", {"source": "A.txt", "chunk_id": 0}),
    Document("FAISS 是用于高维向量相似度搜索的库，可以通过 IndexFlatIP 进行精确内积搜索。", {"source": "B.txt", "chunk_id": 0}),
    Document("CUDA error 1175 是 GPU/CUDA 环境中的一个具体错误，需要根据错误日志进行排查。", {"source": "C.txt", "chunk_id": 0}),
    Document("Transformer 使用 Self-Attention 机制建模不同 Token 之间的关系。", {"source": "D.txt", "chunk_id": 0}),
]
QUERIES = ["什么是 RAG？", "CUDA error 1175 怎么解决？"]
ALPHA = 0.7
TOP_K = 4  # 展示全部 4 篇文档，便于观察每一路归一化时的分数范围。


def number(score: float | None) -> str:
    return "None" if score is None else f"{score:.6f}"


def main() -> None:
    print("model=BAAI/bge-small-zh-v1.5, alpha=0.7, top_k=4")
    print("Vector=IndexFlatIP + normalized embeddings; lexical=BM25L")
    print("Documents（每篇短文直接作为一个检索单元）:")
    for document in DOCUMENTS:
        print(f"{document.metadata['source']}: {document.text}")

    embedder = TextEmbedder()
    faiss = FaissRetriever(DOCUMENTS, embedder.embed_documents(DOCUMENTS))
    vector = TextRetriever(faiss, embedder)
    bm25 = BM25Retriever(DOCUMENTS)
    hybrid = HybridRetriever(vector, bm25, alpha=ALPHA)
    print(f"BM25L: k1={bm25.index.k1}, b={bm25.index.b}, delta={bm25.index.delta}")
    print(f"Document token lengths={bm25.index.doc_len}, avgdl={bm25.index.avgdl:.6f}")

    for query in QUERIES:
        print(f"\nQuery: {query}")
        tokens = tokenize(query)
        print(f"Query tokens: {tokens}")
        vector_results = vector.search(query, top_k=TOP_K)
        bm25_results = bm25.search(query, top_k=TOP_K)
        hybrid_results = hybrid.search(query, top_k=TOP_K)

        print("\nVector Retriever Top-K")
        print("rank | document | raw vector score")
        for result in vector_results:
            print(f"{result.rank} | {result.document.metadata['source']} | {number(result.score)}")

        print("\nBM25 Retriever Top-K（仅返回原始分数 > 0 的结果）")
        print("rank | document | raw BM25 score")
        for result in bm25_results:
            print(f"{result.rank} | {result.document.metadata['source']} | {number(result.score)}")

        print("\nHybrid Retriever Top-K")
        print("rank | document | raw vector | normalized vector | raw BM25 | normalized BM25 | final")
        for result in hybrid_results:
            print(" | ".join([
                str(result.rank), result.document.metadata["source"],
                number(result.vector_score), number(result.normalized_vector_score),
                number(result.bm25_score), number(result.normalized_bm25_score),
                number(result.final_score),
            ]))

        print("\nNormalization ranges（实际返回集）:")
        for name, results in [("Vector", vector_results), ("BM25", bm25_results)]:
            scores = [result.score for result in results]
            if scores:
                print(f"{name}: min={min(scores):.6f}, max={max(scores):.6f}")
            else:
                print(f"{name}: empty")

        # 使用已构建的 BM25L 索引逐词计算贡献，仅用于解释，不参与 Hybrid 排序。
        print("\nBM25L per-token contributions（各项相加得到 raw BM25）:")
        for term in tokens:
            contributions = bm25.index.get_scores([term])
            matches = [
                f"{document.metadata['source']}={score:.6f}"
                for document, score in zip(DOCUMENTS, contributions) if score > 0
            ]
            print(f"{term!r}: idf={bm25.index.idf.get(term, 0):.6f}; " + (", ".join(matches) or "no match"))

        vector_order = [result.document.metadata["source"] for result in vector_results]
        hybrid_order = [result.document.metadata["source"] for result in hybrid_results]
        print(f"\nVector order: {vector_order}")
        print(f"Hybrid order: {hybrid_order}")
        print(f"排序是否改变: {vector_order != hybrid_order}")
        print("None 表示该侧未返回结果，归一化贡献为 0；分数不是正确率。")


if __name__ == "__main__":
    main()
