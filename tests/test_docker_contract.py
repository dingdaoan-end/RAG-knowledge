"""Docker 不可用时的静态契约检查，不能替代真实 build/up。"""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_image_has_valid_single_process_factory_and_non_root_user():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim-bookworm" in dockerfile
    command = json.loads(next(line[4:] for line in dockerfile.splitlines() if line.startswith("CMD ")))
    assert command[:2] == ["uvicorn", "app.deployment:create_deployment_app"]
    assert "--factory" in command and command[command.index("--workers") + 1] == "1"
    assert command[command.index("--host") + 1] == "0.0.0.0" and "--reload" not in command
    assert "USER 10001:10001" in dockerfile and "EXPOSE 8000" in dockerfile
    assert "libgomp1" in dockerfile and "libstdc++6" in dockerfile
    assert "COPY . " not in dockerfile and "ARG API_KEY" not in dockerfile


def test_compose_runtime_paths_separate_uploads_from_readonly_corpus_and_cache():
    services = compose()["services"]
    assert set(services) == {"rag-api", "frontend"}
    service = services["rag-api"]
    assert service["ports"] == ["127.0.0.1:8000:8000"] and service["restart"] == "unless-stopped"
    assert service["env_file"] == [".env"]
    env = service["environment"]
    assert env["UPLOAD_DIR"] == "/app/data/uploads"
    assert env["KNOWLEDGE_FILE"] == "/app/knowledge/knowledge.txt"
    assert env["HF_HUB_CACHE"] == "/models/hub"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert not {"API_KEY", "OPENAI_API_KEY", "MODEL_NAME", "BASE_URL"} & env.keys()
    mounts = {item["target"]: item for item in service["volumes"]}
    assert set(mounts) == {"/app/data/uploads", "/app/knowledge", "/models/hub"}
    assert not mounts["/app/data/uploads"].get("read_only", False)
    assert mounts["/app/knowledge"]["read_only"] and mounts["/models/hub"]["read_only"]
    assert all(item["bind"]["create_host_path"] is False for item in mounts.values())
    assert "./data/uploads" in mounts["/app/data/uploads"]["source"]
    assert "HF_CACHE_DIR" in mounts["/models/hub"]["source"]


def test_both_models_resolve_inside_whole_hub_mount():
    env = compose()["services"]["rag-api"]["environment"]
    assert env["EMBEDDING_MODEL_PATH"].startswith("/models/hub/models--BAAI--bge-small-zh-v1.5/snapshots/")
    assert env["RERANKER_MODEL_PATH"].startswith("/models/hub/models--BAAI--bge-reranker-base/snapshots/")
    assert "EMBEDDING_REVISION" in env["EMBEDDING_MODEL_PATH"] and "RERANKER_REVISION" in env["RERANKER_MODEL_PATH"]


def test_healthcheck_only_requests_local_ready_endpoint():
    service = compose()["services"]["rag-api"]
    test = service["healthcheck"]["test"]
    assert test[:3] == ["CMD", "python", "-c"]
    assert "http://127.0.0.1:8000/ready" in test[3] and "timeout=" in test[3]
    assert "generate" not in test[3] and "embedding" not in test[3]
    assert "HEALTHCHECK" in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_build_context_is_allowlisted_and_excludes_secrets_uploads_and_models():
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "**" in patterns and "!app/**" in patterns
    for pattern in (".env", ".env.*", ".venv/", ".git/", "data/", "results/", "models/", ".cache/"):
        assert pattern in patterns
    assert "!.env" not in patterns and "!data/" not in patterns


def test_example_contains_placeholders_without_overriding_existing_provider():
    # 只读取受版本管理的示例；绝不读取真实 .env。
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    assert values["API_KEY"] == "replace_with_existing_api_key"
    assert values["OPENAI_API_KEY"] == ""
    assert values["MODEL_NAME"] == "replace_with_existing_model_name"
    assert values["BASE_URL"] == "https://replace-with-existing-provider-base-url"
    assert "HF_CACHE_DIR" in values


def test_cpu_dependencies_are_pinned_without_model_download_in_build():
    content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in content
    assert "torch==2.14.0+cpu" in content
    assert "snapshot_download" not in content and "SentenceTransformer(" not in content and "hf download" not in content
    dependencies = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dependencies and "transformers==4.57.6" in dependencies
