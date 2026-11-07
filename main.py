import argparse
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.upload_limits import UploadLimitMiddleware
from app.api.middleware import RequestTraceMiddleware, unhandled_error_response
from app.observability import configure_logging, stage
from app.generation.llm import MockLLM, OpenAILLM
from app.ingestion.loader import load_markdown, load_txt, load_pdf
from app.ingestion.service import IngestionService
from app.ingestion.splitter import character_split, paragraph_split, recursive_split
from app.retrieval.retriever import FaissRetriever
from app.retrieval.text_retriever import TextRetriever
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL, MockReranker
from app.query.rewrite import MockQueryRewriter
from app.pipeline import RAGPipeline
from config import Settings, DeploymentConfigurationError


DATA_PATH = Path(__file__).parent / "data" / "knowledge.txt"
SPLITTERS = {
    "character": character_split,
    "paragraph": paragraph_split,
    "recursive": recursive_split,
}


def build_pipeline(args, settings: Settings, *, embedder=None) -> RAGPipeline:
    suffix = args.file.suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
        raise ValueError("只支持 .txt、.md、.markdown 和 .pdf 文件")
    if not args.file.exists():
        documents = []
    elif suffix == ".pdf":
        documents = load_pdf(args.file)
    else:
        documents = [load_txt(args.file) if suffix == ".txt" else load_markdown(args.file)]

    split = SPLITTERS[args.splitter]
    chunks = []
    for document in documents:
        for chunk in split(document, args.chunk_size, args.chunk_overlap):
            chunk.metadata["chunk_id"] = len(chunks)
            chunks.append(chunk)

    llm = MockLLM() if args.mock else OpenAILLM(settings=settings)
    try:
        reranker = (
            CrossEncoderReranker(model_name=args.reranker_model)
            if args.reranker == "cross-encoder" else MockReranker()
        )
        if getattr(args, "query", None) is None:
            # HTTP 入库维护两套索引；即使查询选 BM25 也预备向量编码器。
            if embedder is None:
                from app.embedding.embedder import TextEmbedder

                embedder = TextEmbedder(model_name=getattr(args, "embedding_model", "BAAI/bge-small-zh-v1.5"), batch_size=args.batch_size)
            pipeline = RAGPipeline(BM25Retriever([]), llm, query_rewriter=MockQueryRewriter(),
                                   reranker=reranker, top_k=args.top_k)
            pipeline.ingestion_service = IngestionService(
                pipeline, embedder, getattr(args, "upload_dir", None) or settings.upload_dir,
                base_chunks=chunks, splitter=split, chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_upload_bytes=getattr(args, "max_upload_bytes", None) or settings.max_upload_bytes,
                candidate_k=args.candidate_k, alpha=args.alpha, retrieval_mode=args.retrieval,
            )
            return pipeline
        if args.retrieval == "bm25":
            retriever = BM25Retriever(chunks, top_k=args.candidate_k)
        else:
            if embedder is None:
                from app.embedding.embedder import TextEmbedder

                embedder = TextEmbedder(model_name=getattr(args, "embedding_model", "BAAI/bge-small-zh-v1.5"), batch_size=args.batch_size)
            chunk_embeddings = embedder.embed_documents(chunks)
            vector_retriever = FaissRetriever(chunks, chunk_embeddings)
            retriever = TextRetriever(vector_retriever, embedder, top_k=args.candidate_k)
            if args.retrieval == "hybrid":
                retriever = HybridRetriever(
                    retriever, BM25Retriever(chunks), alpha=args.alpha, top_k=args.candidate_k,
                )
        return RAGPipeline(
            retriever, llm, query_rewriter=MockQueryRewriter(),
            reranker=reranker, top_k=args.top_k,
        )
    except Exception:
        llm.close()
        raise


def create_app(
    pipeline: RAGPipeline | None = None, ingestion_service: IngestionService | None = None,
    *, pipeline_factory: Callable[[], RAGPipeline] | None = None,
) -> FastAPI:
    if (pipeline is None) == (pipeline_factory is None):
        raise ValueError("必须且只能提供 pipeline 或 pipeline_factory")
    configure_logging()

    async def initialize(app):
        try:
            with stage("initialization"):
                built = await asyncio.to_thread(pipeline_factory)
            app.state.ingestion_service = getattr(built, "ingestion_service", None)
            app.state.pipeline = built
            app.state.initialization_status = "ready"
        except Exception as error:
            # 不把异常正文、配置值或内部堆栈返回给客户端。
            app.state.initialization_detail = (str(error) if isinstance(error, DeploymentConfigurationError)
                                               else "应用初始化失败，请检查本地模型、知识库和上传目录配置。")
            app.state.initialization_status = "failed"

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if pipeline_factory is None:
            app.state.initialization_status = "ready"
        else:
            # 模型和索引加载不阻塞健康接口，加载期间业务路由返回 503。
            task = asyncio.create_task(initialize(app))
        try:
            yield
        finally:
            if task is not None:
                await task
            app.state.initialization_status = "stopped"
            if app.state.pipeline is not None:
                app.state.pipeline.llm.close()

    app = FastAPI(title="RAG Knowledge Base", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware)
    # 追踪中间件在外层，早期 413 也带同一请求 ID。
    app.add_middleware(RequestTraceMiddleware)
    app.add_exception_handler(Exception, unhandled_error_response)
    app.state.pipeline = pipeline
    app.state.ingestion_service = ingestion_service or getattr(pipeline, "ingestion_service", None)
    app.state.initialization_status = "initializing"
    app.state.initialization_detail = ""
    app.include_router(health_router)
    app.include_router(router)
    app.include_router(documents_router)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Knowledge Base API")
    parser.add_argument("--file", type=Path, default=DATA_PATH)
    parser.add_argument("--splitter", choices=SPLITTERS, default="recursive")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5", help="Embedding 模型名称或本地目录")
    parser.add_argument("--upload-dir", type=Path, help="上传目录；默认读取 UPLOAD_DIR 或 data/uploads")
    parser.add_argument("--max-upload-bytes", type=int, help="最大文件字节数；默认 MAX_UPLOAD_BYTES 或 10 MiB")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10, help="重排前召回的候选数")
    parser.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.7, help="融合分数中的向量权重，范围 [0, 1]")
    parser.add_argument("--reranker", choices=["mock", "cross-encoder"], default="mock")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--mock", action="store_true", help="使用 MockLLM，无需 API Key")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--query", help="单次命令行问答；省略此项则启动 API")
    args = parser.parse_args()
    if args.top_k <= 0 or args.candidate_k < args.top_k:
        parser.error("top-k 必须大于 0，且 candidate-k 不能小于 top-k")
    if not 0 <= args.alpha <= 1:
        parser.error("alpha 必须在 [0, 1] 内")
    if args.max_upload_bytes is not None and args.max_upload_bytes <= 0:
        parser.error("max-upload-bytes 必须大于 0")
    pipeline = build_pipeline(args, Settings.from_env())
    if args.query is not None:
        try:
            print(pipeline.ask(args.query))
        finally:
            pipeline.llm.close()
    else:
        uvicorn.run(create_app(pipeline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
