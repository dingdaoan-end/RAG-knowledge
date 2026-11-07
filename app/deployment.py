"""容器专用无参工厂，复用现有应用、配置和 Pipeline 组装。"""

from argparse import Namespace
import os
from pathlib import Path

from config import DeploymentConfigurationError, Settings
from main import build_pipeline, create_app


def _model_path(variable: str, kind: str) -> str:
    value = os.environ.get(variable, "").strip()
    path = Path(value)
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if kind == "embedding":
        required += ["modules.json", "1_Pooling/config.json"]
    if (not value or not path.is_dir() or not all((path / name).is_file() for name in required)
            or not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))):
        raise DeploymentConfigurationError(kind)
    return str(path.resolve())


def deployment_args() -> Namespace:
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise DeploymentConfigurationError("offline")
    return Namespace(
        file=Path(os.environ.get("KNOWLEDGE_FILE", "/app/knowledge/knowledge.txt")),
        embedding_model=_model_path("EMBEDDING_MODEL_PATH", "embedding"),
        reranker_model=_model_path("RERANKER_MODEL_PATH", "reranker"),
        splitter="recursive", chunk_size=300, chunk_overlap=50, batch_size=32,
        top_k=3, candidate_k=10, retrieval="hybrid", alpha=0.7,
        reranker="cross-encoder", mock=False, query=None, upload_dir=None, max_upload_bytes=None,
    )


def build_deployment_pipeline():
    settings = Settings.from_env()
    args = deployment_args()
    # 要求显式沿用已有 Provider 配置，不能落入 SDK 的默认服务或模型。
    if not settings.api_key or not settings.base_url or "MODEL_NAME" not in os.environ:
        raise DeploymentConfigurationError("provider")
    pipeline = build_pipeline(args, settings)
    try:
        # 原 Reranker 延迟加载；正式部署在 ready 前确认真实模型可用，不计算分数。
        pipeline.reranker._get_scorer()
    except Exception:
        pipeline.llm.close()
        raise
    return pipeline


def create_deployment_app():
    return create_app(pipeline_factory=build_deployment_pipeline)
