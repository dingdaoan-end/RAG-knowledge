"""部署装配测试：伪模型文件和 builder，绝不加载权重或发起 API 请求。"""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from config import Settings
from main import create_app
from test_health import empty_pipeline, wait_for_status


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    for name in ("embedding", "reranker"):
        path = tmp_path / name
        path.mkdir()
        for file in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
            (path / file).write_text("{}", encoding="utf-8")
    embedding = tmp_path / "embedding"
    (embedding / "modules.json").write_text("[]", encoding="utf-8")
    (embedding / "1_Pooling").mkdir()
    (embedding / "1_Pooling/config.json").write_text("{}", encoding="utf-8")
    env = {"EMBEDDING_MODEL_PATH": str(embedding), "RERANKER_MODEL_PATH": str(tmp_path / "reranker"),
           "KNOWLEDGE_FILE": str(tmp_path / "knowledge/knowledge.txt"), "UPLOAD_DIR": str(tmp_path / "uploads"),
           "MODEL_NAME": "configured-deepseek-model", "API_KEY": "fake-deployment-key",
           "BASE_URL": "https://configured-provider.invalid", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return env


def deployment():
    return importlib.import_module("app.deployment")


def test_arguments_preserve_algorithms_and_explicit_local_models(local_config):
    args = deployment().deployment_args()
    assert args.embedding_model == local_config["EMBEDDING_MODEL_PATH"]
    assert args.reranker_model == local_config["RERANKER_MODEL_PATH"]
    assert args.file == Path(local_config["KNOWLEDGE_FILE"])
    assert args.reranker == "cross-encoder" and args.mock is False and args.query is None
    assert args.retrieval == "hybrid" and args.alpha == .7 and args.candidate_k == 10 and args.top_k == 3


@pytest.mark.parametrize("variable, model", [
    ("EMBEDDING_MODEL_PATH", "BAAI/bge-small-zh-v1.5"), ("RERANKER_MODEL_PATH", "BAAI/bge-reranker-base"),
])
def test_missing_model_has_explicit_error_and_never_builds(local_config, monkeypatch, variable, model):
    module = deployment()
    monkeypatch.setenv(variable, "/nonexistent-model")
    monkeypatch.setattr(module, "build_pipeline", lambda *a, **kw: pytest.fail("must not initialize models"))
    with pytest.raises(ValueError, match=model):
        module.build_deployment_pipeline()


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors", "tokenizer.json", "modules.json", "1_Pooling/config.json"])
def test_incomplete_embedding_is_rejected_without_fallback(local_config, filename):
    (Path(local_config["EMBEDDING_MODEL_PATH"]) / filename).unlink()
    with pytest.raises(ValueError, match="BAAI/bge-small-zh-v1.5"):
        deployment().deployment_args()


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_online_model_loading_is_rejected(local_config, monkeypatch, variable):
    monkeypatch.setenv(variable, "0")
    with pytest.raises(ValueError, match="离线"):
        deployment().deployment_args()


@pytest.mark.parametrize("variable", ["API_KEY", "MODEL_NAME", "BASE_URL"])
def test_incomplete_provider_cannot_silently_switch_to_defaults(local_config, monkeypatch, variable):
    monkeypatch.delenv(variable)
    module = deployment()
    monkeypatch.setattr(module, "build_pipeline", lambda *a, **kw: pytest.fail("must not build without existing provider config"))
    with pytest.raises(ValueError):
        module.build_deployment_pipeline()


def test_builder_uses_settings_from_env_and_preloads_real_reranker_without_scoring(local_config, monkeypatch):
    module = deployment()
    calls = []
    pipeline = empty_pipeline()
    pipeline.reranker = SimpleNamespace(_get_scorer=lambda: calls.append("load-reranker"))
    original = Settings.from_env

    def read_settings(cls):
        calls.append("settings")
        return original()

    def build(args, settings):
        calls.append("build")
        assert settings.model_name == local_config["MODEL_NAME"]
        assert settings.base_url == local_config["BASE_URL"]
        assert settings.api_key == local_config["API_KEY"]
        assert settings.upload_dir == Path(local_config["UPLOAD_DIR"])
        assert args.mock is False and args.reranker == "cross-encoder"
        return pipeline

    monkeypatch.setattr(Settings, "from_env", classmethod(read_settings))
    monkeypatch.setattr(module, "build_pipeline", build)
    assert module.build_deployment_pipeline() is pipeline
    assert calls == ["settings", "build", "load-reranker"]
    assert pipeline.llm.closed == 0


def test_reranker_loading_failure_closes_provider_and_does_not_fallback(local_config, monkeypatch):
    module = deployment()
    pipeline = empty_pipeline()

    def broken():
        raise OSError("private-model-path")

    pipeline.reranker = SimpleNamespace(_get_scorer=broken)
    monkeypatch.setattr(module, "build_pipeline", lambda *a, **kw: pipeline)
    with pytest.raises(OSError):
        module.build_deployment_pipeline()
    assert pipeline.llm.closed == 1


def test_deployment_factory_is_lazy_and_uses_standard_app(local_config, monkeypatch):
    module = deployment()
    pipeline = empty_pipeline()
    called = []
    monkeypatch.setattr(module, "build_deployment_pipeline", lambda: called.append(True) or pipeline)
    app = module.create_deployment_app()
    assert called == []
    with TestClient(app) as client:
        assert wait_for_status(client, "ready").status_code == 200
        assert client.get("/health").status_code == 200
        assert "/documents/upload" in client.get("/openapi.json").json()["paths"]
    assert called == [True] and pipeline.llm.closed == 1


def test_model_failure_exposes_only_safe_actionable_readiness_detail(local_config, monkeypatch):
    module = deployment()
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "/nonexistent-model")
    with TestClient(create_app(pipeline_factory=module.build_deployment_pipeline)) as client:
        response = wait_for_status(client, "failed")
    assert response.status_code == 503
    assert "BAAI/bge-small-zh-v1.5" in response.json()["detail"]
    assert local_config["API_KEY"] not in response.text and "/nonexistent-model" not in response.text


def test_main_uses_explicit_embedding_path_without_importing_real_model(tmp_path, monkeypatch):
    import main
    from test_ingestion_service import TinyEmbedder
    from test_upload_startup import args

    received = []
    module = ModuleType("app.embedding.embedder")

    def load(**kwargs):
        received.append(kwargs)
        return TinyEmbedder()

    module.TextEmbedder = load
    monkeypatch.setitem(sys.modules, "app.embedding.embedder", module)
    cli = args(tmp_path)
    cli.embedding_model = str(tmp_path / "prepared-embedding")
    pipeline = main.build_pipeline(cli, Settings())
    assert received == [{"model_name": cli.embedding_model, "batch_size": 32}]
    assert pipeline.ingestion_service.snapshot.documents == ()
