from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

import main
from config import Settings
from test_ingestion_service import TinyEmbedder


def args(tmp_path):
    return Namespace(file=tmp_path / "base.txt", splitter="recursive", chunk_size=300, chunk_overlap=50,
                     batch_size=32, top_k=3, candidate_k=10, retrieval="hybrid", alpha=0.7,
                     reranker="mock", reranker_model="unused", mock=True, query=None,
                     upload_dir=tmp_path / "uploads", max_upload_bytes=1024 * 1024)


@pytest.mark.parametrize("base_content", [None, "", " \n\t", "原始基础知识"])
def test_actual_startup_builds_empty_or_original_and_recovers_uploads(tmp_path, base_content):
    # 使用真实 main 组装与实际双索引，仅隔离模型加载。
    cli = args(tmp_path)
    if base_content is not None:
        cli.file.write_text(base_content, encoding="utf-8")
    pipeline = main.build_pipeline(cli, Settings(), embedder=TinyEmbedder())
    with TestClient(main.create_app(pipeline)) as client:
        before = client.post("/chat", json={"question": "问题"}).json()
        assert before["status"] == ("answered" if base_content and base_content.strip() else "no_evidence")
        upload = client.post("/documents/upload", files={"file": ("new.md", "甲星重启保留".encode())})
        assert upload.status_code == 201
    restored = main.build_pipeline(cli, Settings(), embedder=TinyEmbedder())
    with TestClient(main.create_app(restored)) as client:
        response = client.post("/chat", json={"question": "甲星"}).json()
        assert response["sources"][0]["document_id"] == upload.json()["document_id"]
        assert "甲星" in response["sources"][0]["content"]
        if base_content and base_content.strip():
            assert len(restored.ingestion_service.snapshot.documents) == 2


def test_upload_settings_validate_limits_and_leave_llm_configuration_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1234")
    monkeypatch.setenv("MODEL_NAME", "fake-deepseek-test")
    monkeypatch.setenv("BASE_URL", "https://example.test")
    settings = Settings.from_env()
    assert settings.upload_dir == tmp_path / "store" and settings.max_upload_bytes == 1234
    assert settings.model_name == "fake-deepseek-test" and settings.base_url == "https://example.test"
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "0")
    with pytest.raises(ValueError):
        Settings.from_env()
