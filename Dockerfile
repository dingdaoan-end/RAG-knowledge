FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/models/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    UPLOAD_DIR=/app/data/uploads \
    KNOWLEDGE_FILE=/app/knowledge/knowledge.txt \
    HOME=/home/rag

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /home/rag --shell /usr/sbin/nologin rag \
    && mkdir -p /home/rag /app/data/uploads /app/knowledge /tmp/huggingface \
    && chown -R 10001:10001 /home/rag /app/data /tmp/huggingface

WORKDIR /app
COPY requirements.txt requirements-docker.txt ./
# 先安装 CPU wheel，避免 requirements 间接拉取 CUDA 运行库；不下载模型。
RUN python -m pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-docker.txt \
    && python -m pip check

COPY --chown=10001:10001 app/ ./app/
COPY --chown=10001:10001 main.py config.py ./

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).close()"]
CMD ["uvicorn", "app.deployment:create_deployment_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
