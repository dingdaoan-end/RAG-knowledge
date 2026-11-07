# Phase 2：Docker 部署与验证记录

日期：2026-11-06（Asia/Shanghai）。

## 交付与真实验证边界

Dockerfile、Compose、模型/知识库/上传挂载、离线入口、健康/就绪、部署说明及离线测试已经实现。当前环境 PATH 和标准 Docker Desktop 安装目录均没有 Docker，**未执行 docker compose config、镜像构建、容器启动或容器重建验收**；不将 YAML 静态检查、本地 Uvicorn 或 fake 模型测试当成容器实测。

## 文件范围

新增：

- `Dockerfile`：Python 3.11 Debian bookworm CPU 镜像；libgomp1/libstdc++6/证书；UID/GID 10001；单 worker 工厂入口；ready 健康检查。
- `docker-compose.yml`：rag-api、本机 127.0.0.1:8000:8000 映射、unless-stopped、运行时 env_file、三组 bind mount、健康检查和关闭宽限。
- `.dockerignore`：构建内容允许清单，排除环境密钥、虚拟环境、Git、上传、模型、评测及历史结果。
- `requirements-docker.txt`：复用原 requirements，额外约束 CPU torch==2.14.0+cpu、transformers==4.57.6、huggingface-hub==0.36.2。
- `app/deployment.py`：无参部署工厂；固定真实 Cross-Encoder 和原 Hybrid 参数；验证本地模型文件、离线开关和显式 Provider 配置，复用 main.build_pipeline，ready 前预加载 Reranker。
- `app/api/health.py`：GET /health、GET /ready，无推理/检索调用。
- `tests/test_health.py`、`tests/test_deployment.py`、`tests/test_docker_contract.py`：34 个新测试。
- 本记录、`phase2_before_sha256.json`、`phase2_validation.json` 和 `superpowers/plans/2026-11-06-docker.md`：初始指纹、执行计划和验收证据。

修改：

- `main.py`：兼容原 create_app(pipeline)，增加后台 Pipeline 工厂生命周期和状态；新增 embedding-model 装配参数，只传入原模型构造器。
- `app/api/routes.py`：后台加载期间普通/SSE 问答在依赖层返回 503。
- `app/api/upload_limits.py`：无入库服务时在 multipart 暂存前返回 503，避免初始化窗口绕过大小保护。
- `config.py`：增加固定、可公开的部署错误信息；Settings.from_env 的原 Provider/模型/密钥读取规则保持不变。
- `.env.example`：只包含变量名、占位符、公共 snapshot 版本和路径示例；未创建或读取真实 .env。
- `README.md`：完整 Docker 准备/启动/健康/上传/普通与 SSE/备份恢复/故障处理说明。

## 启动与就绪设计

镜像使用 `uvicorn app.deployment:create_deployment_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log`，不使用 main:app，不开启 reload。原 CLI 默认 MockReranker 和字符串问答契约不变；正式部署入口明确设置真实 Cross-Encoder。

后台线程完成配置检查、模型与索引初始化。初始化中 ready 返回 503/initializing，问答/上传返回 503；完成返回 200/ready，空知识库也可 ready；不可恢复的初始化错误保持 503/failed。模型缺失返回固定模型名称/配置项提示，其他异常不返回任意异常正文。关闭时等待初始化任务结束并关闭 Provider 资源。

health 只表示 HTTP 进程存活；ready 只表示本地应用已初始化，均不验证上游密钥、余额或网络，不调用 LLM。两个端点均为 async def，不等待慢问答占满的 AnyIO 工作线程池。当前 Docker restart 策略不会因 unhealthy 自动重启，修复配置后需要手动 restart。

## 模型与数据

检查了宿主机 `C:/Users/吴志武/.cache/huggingface/hub` 的目录与所需文件存在性，没有加载权重：

| 模型 | 已存在的 snapshot | 文件预检 |
|---|---|---|
| BAAI/bge-small-zh-v1.5 | 7999e1d3359715c523056ef9478215996d62a620 | 通过 |
| BAAI/bge-reranker-base | 2cfc18c9415c912f9d8155881c133215df768a70 | 通过 |

Compose 把整个 hub 只读挂载至 /models/hub，显式模型路径指向相应 snapshot，保持 snapshots 与 blobs 引用关系。HF_HOME 指向可写临时目录，HF_HUB_CACHE 指向只读 hub；运行前设置离线/关闭遥测变量。环境变量在 Hub 导入前生效的约束依据 [Hugging Face 官方说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。

CPU torch 版本与本地测试环境的 2.14.0 对齐，使用官方 CPU wheel 源；检查了 [官方 wheel 索引](https://download.pytorch.org/whl/cpu/torch/) 中对应 Python 3.11 Linux x86_64 wheel 的存在性，没有下载或安装到本机。尚未证明镜像内整套依赖安装成功，构建时的 pip check 和容器实测仍需执行。

| 宿主机来源 | 容器路径 | 用途 |
|---|---|---|
| UPLOAD_DATA_DIR，默认 ./data/uploads | /app/data/uploads | 读写上传原件与 manifest |
| KNOWLEDGE_DIR，默认 ./data | /app/knowledge | 只读，默认加载 knowledge.txt |
| HF_CACHE_DIR，用户已有 hub 路径 | /models/hub | 只读本地模型 |

不复制 data/results/models/.env/.venv/.git 进镜像。不持久化 FAISS 二进制；重建时沿用 Phase 1 的大小、SHA256 和切块配置校验。bind mount 禁止自动创建不存在的源路径，避免错误路径生成空目录；相应配置依据 [Docker Compose 官方服务说明](https://docs.docker.com/reference/compose-file/services/)。

## 自动化测试结果

先观察健康接口 7 项、部署模块 17 项、Docker 配置 7 项的缺失行为失败，再完成实现。其中初始化期上传先读 body 的回归观察到 400，修复后在读取前返回 503。额外验证实际 main 将本地 embedding 路径交给原模型构造器。独立审查发现线程池占满导致探针超时；新增回归先失败为 TimeoutError，两个状态端点改成异步后通过。Compose 的端口映射还按审查限定到本机，容器内部继续监听 0.0.0.0。所有用例隔离推理和网络，不下载模型。

| 运行组 | 通过 | 失败 | 跳过 | 耗时 |
|---|---:|---:|---:|---:|
| 新增三文件及上传/启动相关测试 | 59 | 0 | 0 | 2.99 s |
| 全量，除下组两个文件 | 599 | 0 | 0 | 44.45 s |
| test_cross_encoder_reranker.py + test_embedder.py | 38 | 0 | 10 | 80.70 s |
| 全量分组总计 | 637 | 0 | 10 | — |

原 603 个非模型测试保持通过，新增 34 项。10 项跳过的是 RUN_MODEL_TESTS 未开启的真实 BGE 集成测试。两个全量组各报告同一条原有 jieba/pkg_resources 弃用警告。相关组与全量有重复，不重复累加。

在根目录实际执行的相关命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q -p no:cacheprovider tests/test_health.py tests/test_deployment.py tests/test_docker_contract.py tests/test_documents_api.py tests/test_upload_startup.py --tb=short
```

全量由 Python 进程先确认 RUN_MODEL_TESTS 未启用，并令 PYTHONUTF8=1 继承给测试子进程，随后分别运行：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --ignore=tests/test_cross_encoder_reranker.py --ignore=tests/test_embedder.py --tb=short
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_cross_encoder_reranker.py tests/test_embedder.py --tb=short
```

Docker 静态测试只验证 YAML 结构、路径、运行参数、允许清单、占位符、非 root 以及健康检查，不等同于 Docker CLI 规范校验。

## 本地进程 Smoke Test（不是容器）

通过真实 Uvicorn、HTTP 客户端和容器工厂，在隔离临时目录中替换 builder，使用 TinyEmbedder、fake Cross-Encoder scorer 和 evidence-echo fake LLM。实际经历：

- health 200，同时初始化未完成的 ready 503。
- 初始化后 ready 200，空库 /chat 返回 no_evidence。
- TXT 上传 201，随后问答 200，sources 含实际上传 document_id。
- 停止并重新创建服务后，同一文件返回 200/duplicate，document_id 保持一致。

本地临时服务已关闭，临时数据已清理。没有调用真实 LLM，没有实际运行真实模型加载；这些结果不代表 Docker 或真实模型质量验收。

## 仍需在 Docker 机器执行

1. 准备模型、可写上传目录和现有 Provider .env，运行 docker compose config --quiet，避免输出展开的密钥。
2. docker compose up -d --build；确认 CPU 依赖、基础镜像与包源可用。
3. 检查 health/ready，确认 Linux/Windows 挂载权限和本地真实模型加载。
4. 上传文档，记录 ID，down 后 up -d，等待 ready；再次上传应返回 duplicate 与同一 ID，可不调用生成 API。

Linux 上传目录须对 UID/GID 10001 可写。当前支持单实例单 worker、CPU Linux/amd64，不是 GPU 镜像。冷启动和退出等待可能需要随知识库规模调整。没有做容器内真实模型或真实 Provider 测试，没有声称镜像已构建成功。

## 保护边界

原始文件指纹基线共 343 项，不读取 .env 等秘密文件；核验结果写入 phase2_validation.json。评测数据、全部历史 results、检索/Embedding/Reranker/生成/评估/入库核心实现保持原字节；main、API 生命周期与配置装配按本轮范围修改。没有修改 Provider 的实际环境配置，没有真实 LLM 调用、模型下载、四方案评估或原知识库覆盖。完成后停止。
