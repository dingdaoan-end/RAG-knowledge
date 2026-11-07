import os
from dataclasses import dataclass, field
from pathlib import Path


class DeploymentConfigurationError(ValueError):
    """可公开的固定启动提示；不包含环境变量的值。"""

    MESSAGES = {
        "embedding": "缺少或不完整的本地模型 BAAI/bge-small-zh-v1.5，请准备完整缓存并检查 EMBEDDING_MODEL_PATH。",
        "reranker": "缺少或不完整的本地模型 BAAI/bge-reranker-base，请准备完整缓存并检查 RERANKER_MODEL_PATH。",
        "offline": "容器部署必须启用 HF_HUB_OFFLINE=1 和 TRANSFORMERS_OFFLINE=1 离线加载。",
        "provider": "请注入现有 MODEL_NAME、BASE_URL 及 API_KEY 或 OPENAI_API_KEY，部署不会切换 Provider 或模型。",
    }

    def __init__(self, code: str):
        super().__init__(self.MESSAGES[code])


@dataclass
class Settings:
    model_name: str = "gpt-4.1-mini"
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    upload_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "uploads")
    max_upload_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
        if not model_name:
            raise ValueError("MODEL_NAME 不能为空")
        api_key = os.getenv("API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("BASE_URL", "").strip()
        upload_dir = os.getenv("UPLOAD_DIR", "").strip()
        try:
            max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        except ValueError:
            raise ValueError("MAX_UPLOAD_BYTES 必须为正整数") from None
        if max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES 必须为正整数")
        options = {"upload_dir": Path(upload_dir)} if upload_dir else {}
        return cls(model_name=model_name, api_key=api_key or None, base_url=base_url or None,
                   max_upload_bytes=max_upload_bytes, **options)
