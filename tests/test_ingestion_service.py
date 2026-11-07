"""实际双索引与文件提交，仅替代慢的模型推理。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from app.generation.llm import BaseLLM
from app.ingestion.document import Document
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import BM25Retriever


class TinyEmbedder:
    fail = False

    def embed_documents(self, documents):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        vectors = [[int(k in d.text) for k in ("甲星", "乙星", "原始")] + [0.1] for d in documents]
        matrix = np.asarray(vectors, dtype="float32").reshape(-1, 4)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True) if len(matrix) else matrix

    def embed_query(self, question):
        return self.embed_documents([Document(question, {})])


class EvidenceLLM(BaseLLM):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return prompt.split("content:\n", 1)[1].split("\n\n", 1)[0]

    def stream_generate(self, prompt):
        answer = self.generate(prompt)
        yield answer[:2]
        yield answer[2:]


def make_service(tmp_path, *, embedder=None, base=True, **kwargs):
    from app.ingestion.service import IngestionService

    pipeline = RAGPipeline(BM25Retriever([]), EvidenceLLM(), top_k=3)
    base_chunks = [Document("原始知识仍然保留。", {"source": "knowledge.txt", "chunk_id": 0})] if base else []
    service = IngestionService(pipeline, embedder or TinyEmbedder(), tmp_path / "uploads", base_chunks=base_chunks, **kwargs)
    return service, pipeline


def test_first_upload_is_immediately_searchable_and_both_indices_share_objects(tmp_path):
    service, pipeline = make_service(tmp_path)
    response = service.upload("alpha.txt", "甲星的编号为甲一。".encode())
    assert response["status"] == "indexed"
    assert response["document_id"]
    assert "甲星" in pipeline.ask("甲星的编号是什么？")
    assert any("原始" in r.document.text for r in pipeline.retriever.search("原始"))
    snapshot = service.snapshot
    assert len(snapshot.documents) == 2
    assert all(a is b for a, b in zip(snapshot.vector.documents, snapshot.bm25.documents))
    assert len(snapshot.vector.documents) == len(snapshot.bm25.documents)
    ids = [d.metadata["chunk_uid"] for d in snapshot.documents]
    assert len(ids) == len(set(ids))


def test_empty_index_has_no_fake_documents_and_does_not_call_llm(tmp_path):
    service, pipeline = make_service(tmp_path, base=False)
    assert service.snapshot.documents == ()
    result = pipeline.ask_with_result("甲星是什么？")
    assert result.status == "no_evidence"
    assert not pipeline.llm.prompts
    service.upload("first.md", "甲星有资料。".encode())
    assert pipeline.ask_with_result("甲星是什么？").status == "answered"


def test_duplicates_use_content_sha_without_reembedding_or_extra_files(tmp_path):
    embedder = TinyEmbedder()
    service, _ = make_service(tmp_path, embedder=embedder)
    first = service.upload("alpha.txt", "甲星独立资料".encode())
    files = {p.name: p.read_bytes() for p in service.upload_dir.iterdir()}
    embedder.fail = True
    second = service.upload("another.md", "甲星独立资料".encode())
    assert second["status"] == "duplicate"
    assert second["document_id"] == first["document_id"]
    assert {p.name: p.read_bytes() for p in service.upload_dir.iterdir()} == files
    assert len(service.snapshot.documents) == 2


def test_accumulated_uploads_and_restart_keep_stable_ids_and_originals(tmp_path):
    service, _ = make_service(tmp_path)
    service.upload("alpha.txt", "甲星内容。".encode())
    service.upload("beta.md", "乙星内容。".encode())
    before = [(d.text, d.metadata.copy()) for d in service.snapshot.documents]
    restored, pipeline = make_service(tmp_path)
    assert [(d.text, d.metadata) for d in restored.snapshot.documents] == before
    assert "甲星" in pipeline.ask("甲星") and "乙星" in pipeline.ask("乙星")
    assert "原始" in pipeline.ask("原始")
    manifest = json.loads((service.upload_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 2
    for record in manifest["documents"]:
        assert {"document_id", "original_filename", "stored_filename", "sha256", "file_type", "size", "uploaded_at", "chunk_count"} <= record.keys()


@pytest.mark.parametrize("failed_stage", ["embedding", "bm25", "manifest"])
def test_failed_upload_keeps_old_snapshot_manifest_and_files(tmp_path, monkeypatch, failed_stage):
    import app.ingestion.service as module

    embedder = TinyEmbedder()
    service, pipeline = make_service(tmp_path, embedder=embedder)
    service.upload("first.txt", "甲星旧数据。".encode())
    before = {p.name: p.read_bytes() for p in service.upload_dir.iterdir()}
    snapshot = service.snapshot
    if failed_stage == "embedding":
        embedder.fail = True
    elif failed_stage == "bm25":
        monkeypatch.setattr(module, "BM25Retriever", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BM25 build failed")))
    else:
        monkeypatch.setattr(service, "_write_manifest", lambda *a: (_ for _ in ()).throw(OSError("manifest failed")))
    with pytest.raises((RuntimeError, OSError)):
        service.upload("next.txt", "乙星新数据。".encode())
    assert service.snapshot is snapshot and pipeline.retriever is snapshot.retriever
    assert {p.name: p.read_bytes() for p in service.upload_dir.iterdir()} == before
    assert all("乙星" not in d.text for d in service.snapshot.documents)
    embedder.fail = False
    assert "甲星" in pipeline.ask("甲星")


def test_queries_continue_using_old_snapshot_while_upload_builds(tmp_path, monkeypatch):
    service, pipeline = make_service(tmp_path)
    entered, release = Event(), Event()
    original = service._build_snapshot

    def delayed(documents):
        entered.set()
        assert release.wait(5)
        return original(documents)

    monkeypatch.setattr(service, "_build_snapshot", delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload = pool.submit(service.upload, "new.txt", "甲星新数据".encode())
        assert entered.wait(5)
        query = pool.submit(pipeline.ask, "原始")
        try:
            assert "原始" in query.result(timeout=2)
        finally:
            release.set()
        upload.result(timeout=5)
    assert "甲星" in pipeline.ask("甲星")


def test_parallel_uploads_have_no_lost_documents_or_duplicate_ids(tmp_path):
    service, _ = make_service(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: service.upload(*item), [("a.txt", "甲星资料".encode()), ("b.md", "乙星资料".encode()), ("c.txt", "甲星资料".encode())]))
    assert sorted(r["status"] for r in results) == ["duplicate", "indexed", "indexed"]
    assert len(service.snapshot.documents) == 3
    restored, _ = make_service(tmp_path)
    assert len(restored.snapshot.documents) == 3
    assert len({d.metadata["chunk_uid"] for d in restored.snapshot.documents}) == 3


def test_untrusted_name_cannot_escape_upload_directory(tmp_path):
    service, _ = make_service(tmp_path)
    result = service.upload("../../knowledge.txt", "甲星资料".encode())
    record = json.loads((service.upload_dir / "manifest.json").read_text(encoding="utf-8"))["documents"][0]
    assert record["original_filename"] == "knowledge.txt"
    assert record["stored_filename"] != "knowledge.txt"
    assert (service.upload_dir / record["stored_filename"]).parent == service.upload_dir
    assert not (tmp_path / "knowledge.txt").exists()
    assert result["document_id"] == record["document_id"]


@pytest.mark.parametrize("damage", ["missing", "bytes", "path", "duplicate"])
def test_corrupt_persistence_fails_explicitly_without_replacing_current_pipeline(tmp_path, damage):
    service, _ = make_service(tmp_path)
    service.upload("a.txt", "甲星资料".encode())
    path = service.upload_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["documents"][0]
    if damage == "missing":
        (service.upload_dir / record["stored_filename"]).unlink()
    elif damage == "bytes":
        (service.upload_dir / record["stored_filename"]).write_bytes(b"changed")
    elif damage == "path":
        record["stored_filename"] = "../knowledge.txt"
    else:
        manifest["documents"].append(record.copy())
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        make_service(tmp_path)
    assert path.read_bytes() == before


def test_chunk_ids_are_unique_across_pdf_pages(tmp_path, monkeypatch):
    import app.ingestion.service as module

    service, _ = make_service(tmp_path, base=False, chunk_size=8, chunk_overlap=1)
    monkeypatch.setattr(module, "parse_upload", lambda content, filename: [Document("甲星第一页资料", {"source": filename, "page": 1}), Document("乙星第二页资料", {"source": filename, "page": 2})])
    service.upload("pages.pdf", b"%PDF-fixture")
    docs = service.snapshot.documents
    assert len({d.metadata["chunk_uid"] for d in docs}) == len(docs)
    assert [d.metadata["chunk_id"] for d in docs] == list(range(len(docs)))
    assert {d.metadata["page"] for d in docs} == {1, 2}


def test_actual_manifest_replace_failure_preserves_old_file_bytes(tmp_path, monkeypatch):
    import app.ingestion.service as module

    service, pipeline = make_service(tmp_path)
    service.upload("old.txt", "甲星旧资料".encode())
    before = {p.name: p.read_bytes() for p in service.upload_dir.iterdir()}
    real_replace = module.os.replace

    def fail_manifest(source, destination):
        if Path(destination) == service.manifest_path:
            raise OSError("atomic replace unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_manifest)
    with pytest.raises(OSError):
        service.upload("new.txt", "乙星未提交资料".encode())
    assert {p.name: p.read_bytes() for p in service.upload_dir.iterdir()} == before
    assert "甲星" in pipeline.ask("甲星")


def test_cleanup_error_after_manifest_commit_cannot_delete_committed_document(tmp_path, monkeypatch):
    service, pipeline = make_service(tmp_path)
    real_unlink = Path.unlink

    def interrupted_cleanup(path, *args, **kwargs):
        if path.name.startswith(".manifest-"):
            raise OSError("temporary cleanup unavailable")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_cleanup)
    uploaded = service.upload("new.txt", "甲星已经提交资料".encode())
    assert uploaded["status"] == "indexed"
    restored, _ = make_service(tmp_path)
    assert len(restored.snapshot.documents) == 2
    assert "甲星" in pipeline.ask("甲星")


def test_smaller_upload_limit_does_not_invalidate_existing_persistence(tmp_path):
    from app.ingestion.loader import DocumentLoadError

    service, _ = make_service(tmp_path, max_upload_bytes=1024)
    service.upload("old.txt", "甲星已保存资料。".encode())
    restored, pipeline = make_service(tmp_path, max_upload_bytes=8)
    assert "甲星" in pipeline.ask("甲星")
    with pytest.raises(DocumentLoadError) as caught:
        restored.upload("new.txt", b"123456789")
    assert caught.value.status_code == 413
