from .retriever import FaissRetriever, SearchResult


class TextRetriever:
    """把问题文本转成向量，再交给 Day 3 的向量检索器。"""

    def __init__(self, vector_retriever: FaissRetriever, embedder, top_k: int = 3) -> None:
        self.vector_retriever = vector_retriever
        self.documents = vector_retriever.documents
        self.embedder = embedder
        self.top_k = top_k

    def search(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        count = self.top_k if top_k is None else top_k
        return self.vector_retriever.search(query_vector, top_k=count)
