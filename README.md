# RAG Knowledge Base

项目不使用 LangChain 或 LlamaIndex。

## Web 知识库助手

React + TypeScript + Vite 前端位于 `frontend/`。支持 PDF、TXT、Markdown 点击/拖拽上传、普通或流式问答，以及每条回答的真实来源展开。桌面使用双栏，手机使用知识库抽屉。

### Docker 启动（推荐）

先按下文 Docker 部署要求准备原有后端 `.env`、本地模型及数据目录。前端不读取这些配置，也不需要 DeepSeek 密钥。首次启动：

```powershell
cd D:\RAGrepository
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

浏览器打开 **http://127.0.0.1:8080/**。Nginx 提供页面，并将同源 `/api/` 请求转发给 `rag-api:8000`；无需额外开放 CORS。原有后端 `8000` 入口和全部数据挂载保持兼容，浏览器正常使用只需访问 `8080`。`/api/health` 检查后端存活，`/api/ready` 检查就绪；前端容器 healthy 仅代表静态页面正常，页面右上角另外显示后端就绪情况。

后端已经运行、只新增或更新前端时：

```powershell
docker compose build frontend
docker compose up -d --no-deps frontend
docker compose logs -f frontend
```

前端采用 Node 22 构建、非 root Nginx 提供静态文件，不挂载 `.env`、上传文件或模型。构建不会调用 LLM，也不会下载模型；需要网络安装 npm 依赖和拉取基础镜像。Nginx 关闭 SSE 响应缓冲及上传请求暂存，上传大小继续由后端校验；通过 Docker DNS 重新解析后端地址以适配后端容器重建。停止使用 `docker compose down`，不要加 `-v`。上传与 Manifest 持久化机制沿用后端。

### 本地前端开发

需要 Node.js 22.12+（当前也验证了 Node 24.14），另一个窗口先启动现有 FastAPI 或保持后端 Docker 服务运行，然后：

```powershell
cd D:\RAGrepository\frontend
npm ci
npm run dev
```

打开 **http://127.0.0.1:5173/**。Vite Proxy 默认转发至 `http://127.0.0.1:8000`，可在启动 Vite 的进程中用 `API_PROXY_TARGET` 指定其他后端。此变量仅供开发服务器使用，不写进浏览器产物。Vite 明确禁用 dotenv 加载；不要创建任何包含密钥的 `VITE_*` 变量。

### 交互与实际接口

- 上传：`POST /api/documents/upload`，`FormData` 字段为 `file`，浏览器自动设置 multipart boundary。HTTP 201 / `indexed` 与 HTTP 200 / `duplicate` 分别显示新入库和已存在状态，文件名和 document_id 均取自后端。文件后缀选择限制不能替代后端校验。
- 普通问答：关闭“流式回答”后，`POST /api/chat`，请求仅含 `{ "question": "..." }`，响应沿用 `answer`、`status`、`sources`。
- 流式问答：`POST /api/chat/stream`，使用 Fetch + ReadableStream，按 UTF-8 增量解码和 SSE 行/事件边界解析，支持分块、CRLF、多行 data。按实际协议先处理 `sources({sources})`、再追加 `token({token})`；`done({})` 正常完成，`done({status:"no_evidence"})` 显示无证据提示，`error({message})` 显示失败并保留部分答案。未收到 done 的断流、无有效文本、非法 JSON 或来源顺序异常均显式报错，不自动重试。
- Enter 发送，Shift+Enter 换行，中文输入法确认不误发；“停止生成”中止本次请求。回答 Markdown 禁用原始 HTML，省略外部图片，避免模型文本触发第三方图片请求。
- 来源默认折叠，包含后端返回的文件名、document_id、chunk_id、页码（如有）、rank、score 和原文。前端不构造引用，不发起第二次检索。评分仅表示排序分值。
- 页面保留多轮消息展示，但每次请求独立检索，**没有新增服务端会话记忆**。刷新会清空聊天；本机最近 20 条上传元数据保存在浏览器中，“清除记录”只清除本机记录，不删除知识库文件，也不代表完整文档列表。

### 前端验证

```powershell
cd D:\RAGrepository\frontend
npm test
npm run build
```

测试使用 fake HTTP/SSE，不需要真实模型或密钥。需要在浏览器中复现模拟回答时，先运行 `node tests/mock-backend.mjs`，另一个窗口设置 `$env:API_PROXY_TARGET="http://127.0.0.1:18001"` 后运行 `npm run dev -- --port 5174`，只在 `http://127.0.0.1:5174` 发送模拟问题。该夹具不连接真实 API，也不会上传文档；问题含“无证据”或“模拟中断”可复现相应状态。正常开发窗口请不要设置该目标。

本轮真实后端联调仅包含存活/就绪和已存在 PDF 的重复上传；问答验证使用明确标注的模拟服务。没有调用真实 LLM，不能将本轮结果当作真实 DeepSeek 回答验收。详细检查记录见 [前端实施与验收记录](docs/frontend_implementation_2026-11-07.md)。

```text
txt / Markdown / 文本型 PDF（启动文件或 HTTP 上传）
      ↓
Document(text, metadata)
      ↓
character / paragraph / recursive splitter
      ↓
Document chunks（保留 metadata + chunk_id）
      ↓
embed_documents / embed_query（batch + L2 归一化）
      ↓
Query Rewrite → Vector + BM25 → 分数归一化与 alpha 融合
      ↓
候选结果 → Reranker → 最终 Top-K（保留 Document、score、rank）
      ↓
Context Builder → Prompt Template → OpenAILLM / MockLLM
      ↓
FastAPI：/chat（完整答案）或 /chat/stream（SSE 文本增量）
```

## 运行

默认读取 `data/knowledge.txt`，使用 recursive splitter 与 hybrid 检索。启动时合并原始知识库和上传 manifest，重建 FAISS/BM25 双索引，随后由 uvicorn 启动 FastAPI；默认使用真实 OpenAILLM。沿用你已经配置好的 Provider、MODEL_NAME、BASE_URL 和 API_KEY，不要为这一步切换模型：

```powershell
cd D:\RAGrepository
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

服务默认监听 `http://127.0.0.1:8000`，打开 `http://127.0.0.1:8000/docs` 可以提交 question。没有密钥时使用 `.\.venv\Scripts\python.exe main.py --mock`：文档 embedding 和 FAISS 仍真实执行，生成阶段返回固定测试答案。

配置入口是 `Settings.from_env()`，当前代码读取进程环境变量；API_KEY 也支持 OPENAI_API_KEY 别名。自定义 BASE_URL 必须支持现有 Provider 使用的 Responses API。此入口没有自动读取 `.env` 的逻辑，本轮未改动密钥加载方式。修改启动文件需要重启；通过上传接口新增的文件立即生效。

普通接口：

```powershell
$body = @{ question = "什么是 RAG？" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

流式接口（另开终端运行）：

```powershell
'{"question":"What is RAG?"}' | curl.exe --noproxy 127.0.0.1 -N "http://127.0.0.1:8000/chat/stream" -H "Content-Type: application/json" --data-binary "@-"
```

普通接口返回 `answer`、`status` 和 `sources`；流式接口先发送 sources，再发送 token，最终发送 done 或 error。完整字段和上传方法见下文。Streaming 的基本实现见 [Day 5 学习说明](docs/day5.md)。

## 上传、恢复与来源引用（Phase 1）

打开 `http://127.0.0.1:8000/docs`，在 `POST /documents/upload` 选择文件；也可以在 PowerShell 使用：

```powershell
curl.exe --noproxy 127.0.0.1 -F "file=@D:/docs/notes.pdf" http://127.0.0.1:8000/documents/upload
```

支持 `.txt`、`.md`、`.markdown`、`.pdf`。文本必须是 UTF-8（接受 BOM）；PDF 必须有有效签名且能提取正文。扫描、空白、加密和损坏 PDF 返回 422；不支持 OCR。非法扩展名返回 415，超限返回 413。Content-Type 不作为格式判断依据。

新文件返回 HTTP 201：

```json
{"status":"indexed","document_id":"服务端生成的稳定ID","filename":"notes.pdf","file_type":".pdf","size":1234,"sha256":"内容指纹","chunk_count":3}
```

相同字节再次上传返回 HTTP 200、`status="duplicate"` 和已有 document_id；仍检查文件内容与新扩展名是否匹配。入库失败返回脱敏错误，响应头保留 `X-Request-ID`。

文件默认保存在 `data/uploads/`，使用 UUID 文件名，不使用客户端路径。manifest 保存原始文件名、SHA256、大小、上传时间和 Chunk 数量。上传时在写锁内建立完整候选双索引，磁盘提交成功后整体替换检索器。构建失败保留旧索引；查询不持有上传写锁。重启根据原始文件和 manifest 校验 SHA256 后重建索引，无需手工再次上传。备份时同时保存上传目录中的文件和 manifest。

可通过环境变量 `UPLOAD_DIR`、`MAX_UPLOAD_BYTES`，或 CLI `--upload-dir`、`--max-upload-bytes` 设置上传目录和单文件上限（默认 10 MiB）。上传请求总字节另限为“文件上限 + 64 KiB multipart 开销”，在解析/暂存前及接收过程中检查，未声明 Content-Length 也受限。建议上传目录使用绝对路径。

原始知识库文件不存在或没有有效正文时，HTTP 服务仍能启动，`/chat` 返回 200、`status="no_evidence"`、`sources=[]`，不调用 LLM。上传第一个文件后即可查询。

普通问答示例结构：

```json
{"answer":"回答正文","status":"answered","sources":[{"document_id":"稳定ID","source":"notes.pdf","chunk_id":0,"content":"本次真正进入 Prompt 的正文","rank":1,"score":0.9,"page":1}]}
```

来源取自重排及最终 Top-K 之后的实际上下文，不二次检索；source 只返回文件名。page 仅在 PDF 等具备页码的来源中出现。score 是当前结果的检索或重排分数，不代表答案正确率。当前上下文预算就是 `--top-k`，尚无额外 token 截断器。

SSE 成功事件顺序是 `sources → token... → done`，sources payload 为 `{"sources":[...]}`。首段有效文本生成失败时返回 HTTP 502，尚不发送 sources；输出开始后失败发送 error，且不发送 done。no_evidence 发送空 sources、固定说明和 `done {"status":"no_evidence"}`，不会调用 LLM。原字符串接口 `ask()`、`stream_ask()` 保持兼容。

本阶段使用单进程服务；不要用多个 uvicorn workers 共享上传目录。上传会重算全部文档的 embedding/索引，适用于当前小型知识库。恢复要求切块配置与 manifest 一致，损坏或缺失持久化文件会明确阻止恢复。硬崩溃可能留下未提交的孤立文件，它们不进入索引；当前没有自动清理任务。CLI `--query` 仍只查询 `--file`，上传恢复用于 HTTP 服务。

CLI 默认仍为 MockReranker。实际展示 Cross-Encoder 时显式使用已有本地模型（下文 Docker 正式入口已固定启用真实精排）：

```powershell
.\.venv\Scripts\python.exe main.py --reranker cross-encoder --candidate-k 10 --top-k 3
```

本轮未下载模型、未发起真实 LLM 请求。完整验证记录见 [Phase 1 记录](docs/phase1_upload.md)。

## Docker 部署（Phase 2）

提供 Linux/amd64、Python 3.11、CPU 运行配置；使用 Docker Desktop 的 Linux containers/WSL2 或 Linux Docker Engine 与 Compose v2。模型和索引加载需要预留数 GB 内存，CPU 冷启动较慢。镜像构建需要访问基础镜像和 Python 包源，运行时模型完全从本地加载；DeepSeek 问答需要访问你已经配置的 Provider。

当前编写环境未检测到可用的 Docker，镜像 build、Compose 实际启动、Linux 挂载权限和容器重建恢复尚未实测。以下为待执行的部署步骤；已完成静态及离线自动化测试，详见 [Phase 2 验证记录](docs/phase2_docker.md)。

### 1. 准备模型与目录

宿主机必须已有两个完整模型：`BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base`。本项目不会在构建或启动中下载模型。挂载 Hugging Face 的整个 **hub 目录**，其结构应类似：

```text
hub/
  models--BAAI--bge-small-zh-v1.5/
    blobs/                      # 缓存实现使用符号链接时必须保留
    snapshots/<embedding revision>/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      modules.json
      1_Pooling/config.json
      ...
  models--BAAI--bge-reranker-base/
    blobs/
    snapshots/<reranker revision>/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      ...
```

不要只复制 snapshots 中的链接，或仅挂载链接所在的子目录，否则权重可能无法读取。复制缓存到另一台机器时须保留完整文件和有效的相对链接。容器只读挂载 hub，不挂载 Hugging Face home 中的登录 token。

在 PowerShell 中准备持久化目录并查看现有版本目录（这些操作不读取密钥）：

```powershell
cd D:\RAGrepository
New-Item -ItemType Directory -Force data/uploads | Out-Null
Get-ChildItem "$env:USERPROFILE/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots" -Directory -Name
Get-ChildItem "$env:USERPROFILE/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots" -Directory -Name
```

本机已发现的版本号为 `7999e1d3359715c523056ef9478215996d62a620` 和 `2cfc18c9415c912f9d8155881c133215df768a70`。如果你的缓存版本不同，将下面的 EMBEDDING_REVISION、RERANKER_REVISION 填成对应目录名。

Linux 上先创建上传目录，并授予容器 UID/GID **10001:10001** 对该目录的读写权限。例如仅针对专用上传目录执行 `sudo chown -R 10001:10001 data/uploads`。知识库及模型目录只需允许该用户读取和遍历，不能依赖容器内 chown 修改只读挂载。

### 2. 配置运行时环境

已有 `.env` 时保留它；没有时才创建模板副本：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在本地编辑文件，沿用现有 DeepSeek 配置：

| 变量 | 填写内容 |
|---|---|
| API_KEY 或 OPENAI_API_KEY | 已有密钥，二选一；不提交仓库 |
| MODEL_NAME | 当前已经验收成功的模型名称 |
| BASE_URL | 当前 Provider 的兼容地址，保留现有值；当前适配器使用 Responses API |
| HF_CACHE_DIR | 宿主机 hub 的绝对路径，如 `C:/Users/你的用户名/.cache/huggingface/hub` |
| EMBEDDING_REVISION / RERANKER_REVISION | 上一步的 snapshot 目录名 |
| UPLOAD_DATA_DIR | 默认 `./data/uploads`，保留 Phase 1 已有文件与 manifest |
| KNOWLEDGE_DIR | 默认 `./data`，只读加载其中的 knowledge.txt |
| MAX_UPLOAD_BYTES | 默认 10485760（10 MiB） |

Windows 路径使用正斜杠，有空格时按 dotenv 规则加引号。示例中的 Provider 和密钥是占位符，需要替换。Compose 的 env_file 将变量注入容器，应用仍通过 `Settings.from_env()` 读取，Python 不额外读取 `.env`。容器入口要求显式配置模型和 BASE_URL，配置不全不会退回 OpenAI 默认服务。不要把密钥放在 Dockerfile、build ARG 或命令行中。

`docker compose config` 会展开密钥；只用安静模式检查，不将展开结果粘贴到日志或报告：

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose logs -f rag-api
```

### 3. 检查服务

```powershell
curl.exe --noproxy 127.0.0.1 http://127.0.0.1:8000/health
curl.exe --noproxy 127.0.0.1 http://127.0.0.1:8000/ready
docker compose ps
```

- `/health`：200，`{"status":"ok"}`；只表示 HTTP 服务存活。
- `/ready`：初始化中 503/initializing；两个模型、知识库和索引初始化完毕后 200/ready；失败时 503/failed，响应不包含密钥或内部堆栈。空知识库允许 ready。停止时状态为 stopped。
- 两个接口均不调用 LLM、Embedding 或 Reranker。背景初始化独立加载模型；ready 不验证上游 API 的凭据、余额或网络。
- 初始化期间问答及上传返回 503；上传在解析 multipart 前拒绝，避免未就绪时暂存大文件。
- Compose 的容器健康检查请求 `/ready`，启动宽限 180 秒；大知识库可调整该值。unhealthy 不等于 Docker 会自动重启，修复配置后手动 restart。

Swagger：`http://localhost:8000/docs`，可以查看健康检查、上传、普通问答和流式接口。正式容器入口为 `uvicorn app.deployment:create_deployment_app --factory ... --workers 1`，没有开发重载；后台初始化和关闭逻辑复用原应用工厂。

宿主机默认只发布 `127.0.0.1:8000:8000`，保持 8000→8000 映射；容器内部仍监听 0.0.0.0。当前接口没有认证，如需局域网演示，应明确控制访问范围后再调整发布地址。health/ready 使用异步状态读取，不等待慢问答占用的工作线程池。

### 4. 上传、普通问答与 SSE

```powershell
curl.exe --noproxy 127.0.0.1 -F "file=@D:/docs/notes.pdf" http://127.0.0.1:8000/documents/upload

$body = @{question = "这篇资料说明了什么？"} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

'{"question":"What does the document explain?"}' | curl.exe --noproxy 127.0.0.1 -N http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" --data-binary "@-"
```

上传支持 TXT/Markdown/文本 PDF，返回 document_id。普通回答保留 answer/status/sources；SSE 先 sources，再 token，最后 done，失败用 error。首次上传后无需重启即可查询。空库或无候选直接 no_evidence，不调用 LLM。执行后两条问答命令会调用真实 Provider 并产生正常 API 用量；只检查部署时使用 health/ready 和上传即可。

正式容器固定启用 Hybrid + **真实 Cross-Encoder**，保留 alpha=0.7、Top-10 候选、Top-3 上下文。Embedding/重排路径由 Compose 指向只读 snapshot，HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、HF_HUB_DISABLE_TELEMETRY=1 在进程启动前生效；缺模型明确报错，不自动改成 Mock。

### 5. 数据持久化、停止与恢复

| 宿主机来源 | 容器路径 | 权限与用途 |
|---|---|---|
| UPLOAD_DATA_DIR | /app/data/uploads | 读写；上传原文件与 manifest.json |
| KNOWLEDGE_DIR | /app/knowledge | 只读；原始知识库，不会由启动覆盖 |
| HF_CACHE_DIR | /models/hub | 只读；完整模型缓存，权重不在镜像中 |

容器删除、重建后仍从同一上传目录恢复。沿用 Phase 1 的大小/SHA256/切块配置校验，重新生成向量和双索引，无需持久化 FAISS 二进制。备份时停止写入并同时备份上传目录的文件与 manifest；只保存其中一部分不能完整恢复。V1/V2/V3 评测数据和历史报告不作为运行时写目录，也不复制进镜像。

```powershell
docker compose restart rag-api
docker compose down
docker compose up -d
```

停止命令不带 `-v`。同一上传目录只运行一个服务实例、一个 worker；不要 scale。

建议的容器恢复验收：上传文件并记下 document_id，执行 `docker compose down` 后再 `up -d`，等待 ready，再上传相同文件应返回 duplicate 及同一 ID；这一步无需生成答案或调用真实 API。

### 常见问题

- 找不到 docker：安装 Docker Engine/Compose v2 或 Docker Desktop 并启动 Linux 引擎，再执行 `docker version`、`docker compose version`。
- bind source 不存在：先创建上传目录，检查 KNOWLEDGE_DIR、HF_CACHE_DIR。Compose 禁止自动创建错误的挂载路径。
- ready 显示缺少 BGE 模型：检查 hub 路径、snapshot 版本、tokenizer、Pooling 及权重是否完整；修正后 restart。不会在线补下载。
- ready 一直 initializing：首次 CPU 模型加载、恢复及全部文档 embedding 尚未结束；结合 initialization 阶段日志检查内存和目录权限。
- ready 失败但 health 正常：进程存活而应用不可用；修复模型、Provider 或损坏的 manifest。不要删除 SHA 校验，也不要覆盖原文件来掩盖问题。
- Permission denied：Linux 宿主机上传目录应可被 UID/GID 10001 写入，模型和知识库应可读。
- Provider 401/超时：ready 不会探测上游 API；检查原环境配置和网络，不把密钥粘贴到日志。
- 8000 端口占用：停止占用服务，或只修改 Compose 的宿主机端口，例如 `127.0.0.1:8001:8000`。
- 不要复制 `.venv`、`.env`、用户上传文件、模型或实验结果进镜像；`.dockerignore` 使用允许清单，Dockerfile 也只复制必要源文件。

Day 6 默认接入 MockQueryRewriter 和 MockReranker：前者只去除首尾空白，后者保持候选顺序，方便以后替换真实 LLM/Cross Encoder。详细实现、分数计算示例和复杂度见 [Day 6 学习说明](docs/day6.md)。

```powershell
# candidate-k：重排前召回数量；top-k：最终送给 LLM 的数量
.\.venv\Scripts\python.exe main.py --mock --retrieval hybrid --alpha 0.7 --candidate-k 10 --top-k 3
# 单次命令行问答；BM25 模式无需加载 embedding 模型
.\.venv\Scripts\python.exe main.py --mock --retrieval bm25 --query "FAISS 是什么？"
# 比较检索模式，评估只测检索，不调用生成 LLM
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval vector
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval bm25
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval hybrid --alpha 0.7
```

读取 Markdown 并选择段落切分：

```powershell
.\.venv\Scripts\python.exe main.py `
  --mock `
  --file "D:\docs\notes.md" `
  --splitter paragraph `
  --chunk-size 300 `
  --chunk-overlap 50 `
  --query "这篇文档讲了什么？"
```

三个 splitter：

- `character`：固定字符窗口，重叠长度最精确。
- `paragraph`：优先组合以空行分隔的完整段落，超长段落退回字符切分。
- `recursive`：按 `"\n\n" → "\n" → "。" → "！" → "？" → "；" → "，"` 递归寻找边界，最后退回字符切分。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

上述命令运行快速测试，跳过需要真实模型的 integration 测试。完整测试：

```powershell
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

真实测试使用 `BAAI/bge-small-zh-v1.5`，首次运行需要下载模型；已有本机缓存时，可设置 `$env:HF_HUB_OFFLINE = "1"` 使用缓存。

## Day 4 的模块职责

- `app/generation/context_builder.py`：`build_context(results)` 接收 SearchResult 或包含 document/score/rank 的字典，按输入顺序拼接所有检索结果。保留 source、chunk_id、rank、score、其他 metadata 和正文，不修改原始 Document。空结果返回“没有检索到可用的知识库资料。”。
- `app/generation/prompt.py`：SYSTEM_PROMPT 集中保存知识库回答规则，`build_prompt(question, context)` 组装 System、Context、Question 三部分。
- `app/generation/llm.py`：BaseLLM 定义 generate 和 stream_generate；MockLLM 用于测试，OpenAILLM 使用官方 SDK 真实生成。
- `app/generation/__init__.py`：导出生成模块的函数和类。
- `app/retrieval/text_retriever.py`：TextRetriever 对外提供 `search(question)`，内部调用 embed_query，再调用原有 FaissRetriever.search(query_vector, top_k)。这是连接文本问题与向量检索的简单适配层。
- `app/retrieval/__init__.py`：增加 TextRetriever 导出。
- `app/pipeline.py`：RAGPipeline 接收 retriever、llm，仅负责四步流程编排。
- `main.py`：创建实际 embedding/向量检索器、TextRetriever、LLM 和 pipeline，默认启动 HTTP 服务；保留 --query 单次问答。
- `tests/test_generation.py`：测试 context、prompt、LLM 接口、文本检索适配、完整调用链和空检索。除固定答案测试外，还用记录 prompt 的测试 LLM 检查问题与检索资料确实传入生成阶段，防止固定答案掩盖流程错误。
- `README.md`：记录运行方式和设计原因。

## Day 4 的基础 Pipeline（Day 6 已扩展改写与重排）

```python
def ask(self, question):
    results = self.retriever.search(question)
    context = build_context(results)
    prompt = build_prompt(question, context)
    return self.llm.generate(prompt)
```

pipeline 不调用 embed_query、不操作 FAISS、不处理模型的网络请求；这些细节分别由检索和 LLM 模块承担。检索器只需提供文本 search 接口，LLM 只需提供 generate 接口。更换检索策略或接入新的语言模型时，只需要替换对应对象，pipeline 可以继续使用相同流程。

```text
建库阶段：
Loader → Document → Splitter → chunks → embed_documents → FAISS 索引

问答阶段：
question
  → TextRetriever.search(question)
      → embed_query(question)
      → FaissRetriever.search(query_vector, top_k)
  → SearchResult 列表
  → build_context(results)
  → build_prompt(question, context)
  → llm.generate(prompt)
  → answer
```

Context 示例：

```text
[Document 1]
source: notes.md
chunk_id: 0
rank: 1
score: 0.85

content:
RAG 是检索增强生成。
```

Prompt Template 的作用是统一回答约束，并明确区分资料和问题。将规则放在独立文件，便于调整提示词和单独测试，不必修改检索或 pipeline。System/Context/Question 是提示词字符串中的段落标签；OpenAILLM 将完整 prompt 作为 Responses API 的 input 传入。

当前 Pipeline 在最终候选为空时直接返回 no_evidence，不调用 LLM。非空上下文是否足够回答仍依赖模型按 Prompt 处理，尚无通用证据充分性分类。MockLLM 返回固定测试答案，用于验证调用链。

## Day 3 的接口

```python
embedder = TextEmbedder(batch_size=32)
document_vectors = embedder.embed_documents(chunks)  # (N, 512)，输入 list[Document]
query_vector = embedder.embed_query("FAISS 有什么用？")  # (1, 512)
retriever = FaissRetriever(chunks, document_vectors)
results = retriever.search(query_vector, top_k=3)
for result in results:
    print(result.rank, result.score, result.document.metadata)
```

`batch_size` 决定模型每次推理处理多少条文本，实际分批由 `SentenceTransformer.encode()` 完成。即使输入数量超过一个 batch，返回矩阵仍按原始顺序包含所有向量。文档和查询都传入 `normalize_embeddings=True`，并返回连续的 float32 数组。

`embed_documents([])` 返回 `(0, 512)`；空白查询报 `ValueError`。Retriever 检查向量维度、数量与有限数值，空索引返回 `[]`，非法 `top_k` 报 `ValueError`，超过文档数量的 `top_k` 被截断。排名从 1 开始，分数按从高到低排列。检索结果引用原始 Document，所以其 metadata 会完整保留。相同分数之间的顺序不作额外保证。

## 为什么内积等于余弦相似度

```text
cos(q, x) = (q · x) / (||q||₂ × ||x||₂)
q̂ = q / ||q||₂
x̂ = x / ||x||₂
q̂ · x̂ = cos(q, x)
```

归一化把非零向量长度变为 1，保留方向。IndexFlatIP 按内积分数降序检索，因此对双方都归一化的向量，它等价于按余弦相似度检索。仅归一化查询不能消除文档向量长度的影响。score 是相似度，不是回答正确的概率。

## 评估 Recall@1、Recall@3、Recall@5

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

可以通过 `--dataset 路径` 指定自己的 JSON，结构参考 `data/retrieval_eval.json`。documents 中的每一条是一个检索单元，id 必须唯一；queries 中的 relevant_ids 是人工标注的相关文档 id。这里直接评估 embedding + 检索，不再切分评估文档，避免切分变化导致标签失效。

```text
单查询 Recall@K = |前 K 个结果的 id 集合 ∩ 相关文档 id 集合| / |相关文档 id 集合|
最终 Recall@K = 所有查询 Recall@K 的算术平均
```

例：相关文档是 `{a, b}`，Top-1 返回 `a`，Recall@1 是 0.5；Top-3 同时找回 a、b，Recall@3 是 1。只命中一条就计 1 属于 Hit Rate，不能用于多相关文档的 Recall。评估每个查询只检索一次 Top-5，再分别使用前 1、3、5 个结果计算。

示例集包含 8 个文档、6 个查询，其中一个查询标注两个相关文档。因此即便每个查询都排序正确，平均 Recall@1 的上限仍是 `(5 × 1 + 0.5) / 6 = 0.9167`。这是一组学习用的微型评估数据，不能代表真实业务效果。

本次用真实 BGE 模型运行示例集：Recall@1 = 0.9167，Recall@3 = 1.0000，Recall@5 = 1.0000。

## 逐文件理解核心实现

- `app/embedding/embedder.py`：提取 Document.text，调用真实 sentence-transformers 模型做分批编码和归一化；查询返回一行向量，空文档返回有正确维度的空矩阵。
- `app/retrieval/retriever.py`：保存 Documents 与向量行号的对应关系，建立 IndexFlatIP，检查输入，调用 FAISS，按行号找回文档并附上 score、rank。SearchResult 只是包含三个字段的 dataclass。
- `app/retrieval/__init__.py`：导出 FaissRetriever 与 SearchResult。
- `main.py`：调用新 embedding 接口，支持 `--batch-size`、`--top-k`；Day 4 通过 RAGPipeline 输出 answer。
- `evaluate_retrieval.py`：加载标注，建立索引，逐查询检索，用集合交集计算召回，再取平均；核心统计函数与模型加载分开，便于手工验证公式。
- `data/retrieval_eval.json`：可直接修改的人工标注样本。
- `tests/test_embedder.py`：用真实 BGE 验证 shape、单位长度、分批前后结果及文本相关性。
- `tests/test_retriever.py`：用手算得出分数的向量验证排序、数量、metadata 与异常边界。
- `tests/test_evaluate_retrieval.py`：用人工可核对的例子验证多标签召回、平均值与空索引。
- `README.md`：记录接口、运行方法、公式与性能限制；Day 2 的 loader/splitter 保持原有功能。

## 复杂度与以后要优化的位置

令 N 为文档块数量，d 为向量维度，K 为返回数量，Q 为评估查询数。

- 向量归一化：O(Nd)；模型推理还取决于 token 长度及模型结构，不能简单写成 O(N)。batch 控制推理过程的内存峰值，但完整输出仍占 O(Nd)。
- Flat 索引建立：O(Nd) 时间、O(Nd) 存储；float32 原始向量约占 4Nd 字节，不含 Python metadata 和构建时输入矩阵。
- 单查询：扫描全部向量，计算内积为 O(Nd)；Top-K 选择还存在开销，小 K 下扫描通常占主导，返回结果构造为 O(K)。
- 评估：Q 次检索主要为 O(QNd)，还需计算查询 embedding 和集合交集。

数据增多后应先缓存文档向量、持久化索引，避免每次运行重新编码；再根据内存和吞吐量调 batch、批量处理查询。Flat 扫描成为瓶颈时评估 HNSW/IVF 等近似索引，用速度与 Recall 的对比决定参数；内存不足时再评估 PQ/量化。与此同时扩大人工标注集，把切分、embedding 和检索质量一起测量。
