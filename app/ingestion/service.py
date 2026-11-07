"""单进程文档入库：磁盘提交成功后发布整套检索快照。"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import Lock
from uuid import uuid4

from app.ingestion.document import Document
from app.ingestion.loader import ALLOWED_EXTENSIONS, DocumentLoadError, display_filename, parse_upload
from app.ingestion.splitter import recursive_split
from app.observability import stage
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.retrieval.retriever import FaissRetriever
from app.retrieval.text_retriever import TextRetriever


@dataclass(frozen=True)
class IndexSnapshot:
    documents: tuple[Document, ...]
    vector: TextRetriever
    bm25: BM25Retriever
    retriever: object


class IngestionService:
    def __init__(
        self, pipeline, embedder, upload_dir: str | Path, *, base_chunks=(),
        splitter=recursive_split, chunk_size: int = 300, chunk_overlap: int = 50,
        max_upload_bytes: int = 10 * 1024 * 1024, candidate_k: int = 10,
        alpha: float = 0.7, retrieval_mode: str = "hybrid",
    ):
        if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
            raise ValueError("最大上传字节数必须是正整数")
        if retrieval_mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("不支持的检索模式")
        # 复用 splitter 的参数检查，空文档不构造虚假 chunk。
        splitter(Document("", {}), chunk_size, chunk_overlap)
        self.pipeline, self.embedder = pipeline, embedder
        self.splitter, self.chunk_size, self.chunk_overlap = splitter, chunk_size, chunk_overlap
        self.max_upload_bytes, self.candidate_k = max_upload_bytes, candidate_k
        self.alpha, self.retrieval_mode = alpha, retrieval_mode
        self.upload_dir = Path(upload_dir).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.upload_dir / "manifest.json"
        self._write_lock = Lock()
        self._chunking = {"splitter": splitter.__name__, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        self._records = self._load_manifest()
        documents = self._base_chunks(base_chunks)
        for record in self._records:
            path = self._stored_path(record["stored_filename"])
            if not path.is_file() or path.stat().st_size != record["size"]:
                raise ValueError("已保存文档缺失或大小不一致，停止恢复")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("已保存文档 SHA256 不一致，停止恢复")
            chunks = self._chunks(parse_upload(content, record["original_filename"]), record["document_id"])
            if len(chunks) != record["chunk_count"]:
                raise ValueError("恢复的 Chunk 数量不一致")
            documents.extend(chunks)
        snapshot = self._build_snapshot(documents)
        self.snapshot = snapshot
        self.pipeline.retriever = snapshot.retriever

    def _base_chunks(self, chunks):
        result = []
        for index, chunk in enumerate(chunks):
            metadata = deepcopy(chunk.metadata)
            original_source = str(metadata.get("source", "knowledge.txt"))
            metadata["source"] = display_filename(original_source)
            document_id = metadata.get("document_id") or "base_" + hashlib.sha256(original_source.encode()).hexdigest()[:32]
            chunk_id = metadata.get("chunk_id", index)
            metadata.update(document_id=document_id, chunk_id=chunk_id, chunk_uid=f"{document_id}:{chunk_id}")
            result.append(Document(chunk.text, metadata))
        return result

    def _chunks(self, documents, document_id):
        chunks = []
        for document in documents:
            metadata = {**document.metadata, "document_id": document_id}
            for chunk in self.splitter(Document(document.text, metadata), self.chunk_size, self.chunk_overlap):
                chunk_id = len(chunks)
                chunk.metadata.update(chunk_id=chunk_id, chunk_uid=f"{document_id}:{chunk_id}")
                chunks.append(chunk)
        if not chunks:
            raise DocumentLoadError("文档没有可用于入库的正文")
        return chunks

    def _build_snapshot(self, documents):
        documents = list(documents)
        ids = [document.metadata["chunk_uid"] for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("文档 Chunk 稳定 ID 重复")
        with stage("ingestion_index"):
            embeddings = self.embedder.embed_documents(documents)
            vector = TextRetriever(FaissRetriever(documents, embeddings), self.embedder, top_k=self.candidate_k)
            bm25 = BM25Retriever(documents, top_k=self.candidate_k)
            # Hybrid 构造本身验证两路对象及顺序，所有模式都建立一致的双索引。
            hybrid = HybridRetriever(vector, bm25, alpha=self.alpha, top_k=self.candidate_k)
            retriever = {"vector": vector, "bm25": bm25, "hybrid": hybrid}[self.retrieval_mode]
            return IndexSnapshot(tuple(documents), vector, bm25, retriever)

    def _stored_path(self, name):
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise ValueError("manifest 包含无效保存路径")
        path = self.upload_dir / name
        if path.is_symlink() or path.resolve().parent != self.upload_dir:
            raise ValueError("保存路径必须位于上传目录内")
        return path

    def _load_manifest(self):
        if self.manifest_path.is_symlink():
            raise ValueError("manifest 不能是符号链接")
        if not self.manifest_path.exists():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload["version"] != 1 or payload["chunking"] != self._chunking:
                raise ValueError("manifest 版本或切块配置不一致")
            records = payload["documents"]
            if not isinstance(records, list):
                raise ValueError("manifest documents 必须是列表")
            seen_ids, seen_hashes = set(), set()
            for r in records:
                document_id, digest, suffix = r["document_id"], r["sha256"], r["file_type"]
                if not re.fullmatch(r"[0-9a-f]{32}", document_id) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("manifest ID 或指纹无效")
                if document_id in seen_ids or digest in seen_hashes:
                    raise ValueError("manifest 文档 ID 或指纹重复")
                seen_ids.add(document_id)
                seen_hashes.add(digest)
                if suffix not in ALLOWED_EXTENSIONS or r["stored_filename"] != document_id + suffix:
                    raise ValueError("manifest 文件名或类型无效")
                if display_filename(r["original_filename"]) != r["original_filename"]:
                    raise ValueError("manifest 展示文件名无效")
                # 新的上传限额不应使先前合法持久化的文件失效。
                if type(r["size"]) is not int or r["size"] <= 0:
                    raise ValueError("manifest 文件大小无效")
                if type(r["chunk_count"]) is not int or r["chunk_count"] <= 0:
                    raise ValueError("manifest Chunk 数量无效")
                if datetime.fromisoformat(r["uploaded_at"]).utcoffset() is None:
                    raise ValueError("manifest 上传时间缺少时区")
            return records
        except (KeyError, TypeError, UnicodeError) as error:
            raise ValueError("manifest 格式不合法，停止恢复") from error

    def _write_manifest(self, records):
        content = json.dumps({"version": 1, "chunking": self._chunking, "documents": records}, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.upload_dir, prefix=".manifest-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
            # replace 已将临时文件移走；提交后不再执行可能失败的清理操作。
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def upload(self, filename: str, content: bytes):
        filename = display_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise DocumentLoadError("仅支持 TXT、Markdown 和文本型 PDF", status_code=415)
        if len(content) > self.max_upload_bytes:
            raise DocumentLoadError("文件超过最大上传大小", status_code=413)
        digest = hashlib.sha256(content).hexdigest()
        with self._write_lock:
            # 去重也必须先通过当前扩展名对应的内容检查。
            parsed = parse_upload(content, filename)
            for record in self._records:
                if record["sha256"] == digest:
                    return self._result(record, "duplicate")
            with stage("ingestion"):
                document_id = uuid4().hex
                chunks = self._chunks(parsed, document_id)
                snapshot = self._build_snapshot([*self.snapshot.documents, *chunks])
                record = {
                    "document_id": document_id, "original_filename": filename,
                    "stored_filename": document_id + suffix, "sha256": digest,
                    "file_type": suffix, "size": len(content),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(), "chunk_count": len(chunks),
                }
                records = [*self._records, record]
                final = self._stored_path(record["stored_filename"])
                temporary, installed, committed = None, False, False
                try:
                    with NamedTemporaryFile(dir=self.upload_dir, prefix=".upload-", suffix=".tmp", delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if final.exists():
                        raise FileExistsError("服务端文件名发生冲突")
                    os.replace(temporary, final)
                    temporary = None
                    installed = True
                    self._write_manifest(records)
                    committed = True
                    self._records = records
                    self.snapshot = snapshot
                    # 查询只取一次 retriever 引用；旧索引从未原地修改。
                    self.pipeline.retriever = snapshot.retriever
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    if installed and not committed:
                        final.unlink(missing_ok=True)
                return self._result(record, "indexed")

    @staticmethod
    def _result(record, status):
        return {
            "status": status, "document_id": record["document_id"],
            "filename": record["original_filename"], "file_type": record["file_type"],
            "size": record["size"], "sha256": record["sha256"], "chunk_count": record["chunk_count"],
        }
