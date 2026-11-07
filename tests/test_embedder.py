import os

import numpy as np
import pytest

from app.ingestion.document import Document


@pytest.fixture(scope="module")
def embedder():
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("设置 RUN_MODEL_TESTS=1 运行真实 BGE 模型测试")
    from app.embedding.embedder import TextEmbedder

    return TextEmbedder(batch_size=2)


@pytest.mark.integration
def test_embedding_shapes_and_empty_documents(embedder):
    documents = [Document("向量检索", {}), Document("公园散步", {})]
    vectors = embedder.embed_documents(documents)
    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    assert embedder.embed_query("什么是向量检索？").shape == (1, 512)
    assert embedder.embed_documents([]).shape == (0, 512)


@pytest.mark.integration
def test_document_and_query_vectors_are_normalized(embedder):
    vectors = embedder.embed_documents([Document("RAG 使用语义检索。", {}), Document("植物通过光合作用生长。", {})])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1], atol=1e-5)
    query = embedder.embed_query("如何检索相关资料？")
    np.testing.assert_allclose(np.linalg.norm(query, axis=1), [1], atol=1e-5)


@pytest.mark.integration
def test_batching_preserves_order_and_embeddings(embedder):
    documents = [Document(text, {}) for text in ["向量检索", "光合作用", "足球比赛", "文本切块", "鸡蛋炒饭"]]
    batched = embedder.embed_documents(documents)
    individually = np.concatenate([embedder.embed_documents([document]) for document in documents])
    assert batched.shape == (5, 512)
    np.testing.assert_allclose(batched, individually, atol=1e-5)


@pytest.mark.integration
def test_real_model_ranks_relevant_text_first(embedder):
    from app.retrieval.retriever import FaissRetriever

    documents = [
        Document("植物通过光合作用将阳光转化为化学能。", {"source": "biology.txt"}),
        Document("FAISS 用于向量检索，能够搜索与查询最相似的向量。", {"source": "faiss.txt"}),
        Document("煮米饭时需要加入适量的水。", {"source": "cooking.txt"}),
    ]
    retriever = FaissRetriever(documents, embedder.embed_documents(documents))
    results = retriever.search(embedder.embed_query("FAISS 是用来做什么的？"), top_k=3)
    assert results[0].document.metadata["source"] == "faiss.txt"
    assert results[0].score > results[1].score


@pytest.mark.integration
@pytest.mark.parametrize("query", ["", "  \n"])
def test_blank_query_is_rejected(embedder, query):
    with pytest.raises(ValueError):
        embedder.embed_query(query)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(embedder, batch_size):
    from app.embedding.embedder import TextEmbedder

    with pytest.raises(ValueError):
        TextEmbedder(batch_size=batch_size)
