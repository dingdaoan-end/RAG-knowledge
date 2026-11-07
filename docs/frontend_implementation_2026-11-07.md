# Web 前端实施与验收记录

## 用户已批准的范围

React + TypeScript + Vite，浅色桌面双栏、移动知识库抽屉。左侧真实上传与本机最近记录，右侧独立问题的多轮消息展示，默认流式、可选择普通回答；真实 sources 折叠展示。前端不读后端密钥，不实现后端文档列表或会话记忆。

接口以实际代码为准：上传 multipart 字段 file；普通结果 answer/status/sources；流式 sources({sources}) → token({token}) → done({} 或 status=no_evidence)，失败 error({message})。首段前错误为 HTTP 502。

## 实施顺序

1. [x] 创建 frontend 工程和协议测试，观察缺失实现失败，再实现 Fetch API 与增量 UTF-8/SSE 解析。
2. [x] 添加组件交互测试，再实现上传、记录、聊天、来源、停止、错误和响应式布局。
3. [x] 增加 Node/Nginx 多阶段镜像与 Compose frontend 服务，保留 rag-api 原配置与挂载；Nginx 关闭 SSE/请求缓冲，动态解析 Docker DNS。
4. [x] 运行 TypeScript/生产构建、前端测试、相关及全量后端回归；浏览器查看桌面/手机布局。
5. [x] 真实后端只联调 health/ready 和已存在 PDF 的重复上传，不调用真实 LLM。模拟回答测试单独标明，保存保护文件 SHA256 和最终结果。

## 技术决策

- API 统一通过同源 /api，开发用 Vite Proxy，生产用 Nginx；不启用宽泛 CORS。
- 上传历史仅保存浏览器元数据，不冒充后端完整知识库列表；回答/正文不持久化。
- SSE 必须先 sources 后 token；连接未 done、非法 JSON、错误事件均显式失败，不自动重连或重复请求。
- 保留实时部分回答；AbortController 支持用户主动停止。
- 测试不读取根目录 .env；Vite 禁用 dotenv 加载，Docker frontend 构建上下文仅为 frontend。
- Nginx 请求体大小交由已存在的后端限额校验，关闭请求暂存；使用 Docker DNS 动态解析，适配后端重建后的 IP 变化。

## 验证结果

### 文件与边界

新增 `frontend/` 工程：package/lock、TypeScript/Vite 配置、index.html、src/main.tsx、src/App.tsx、src/Sources.tsx、src/api.ts、src/styles.css、两组单元测试与 setup、tests/mock-backend.mjs、Dockerfile、nginx.conf、.dockerignore。

已有文件仅修改四个：`docker-compose.yml`（追加 frontend）、`.gitignore`（排除前端依赖与产物）、`README.md`（使用说明）、`tests/test_docker_contract.py`（已有服务集合断言增加 frontend，不删除 rag-api 检查）。新增 `tests/test_frontend_deployment.py` 检查四项部署边界。所有后端 app 源码、config.py、main.py、requirements.txt 和检索算法均未修改。

本报告及 `frontend_before_sha256_2026-11-07.json`、`frontend_backend_service_before.json`、`frontend_integrity_2026-11-07.json`、`frontend_container_validation_2026-11-07.json`、截图为本轮新增验收材料。仓库原文件尚未形成可用 Git 基线，所以用修改前后 SHA256 和原 rag-api YAML 字典比对；不以空 Git diff 作为无变更证明。

### 自动化结果（2026-11-07）

| 命令/范围 | 通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: |
| `cd frontend; npm test -- --reporter=dot` | 37 | 0 | 0 |
| Python 全量第一组，排除 test_cross_encoder_reranker.py 和 test_embedder.py | 603 | 0 | 0 |
| Python 全量第二组，仅以上两个文件 | 38 | 0 | 10 |
| **后端全量合计（无交集）** | **641** | **0** | **10** |

Python 命令使用 `PYTHONUTF8=1` 及项目 `.venv`：

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q --ignore=tests/test_cross_encoder_reranker.py --ignore=tests/test_embedder.py
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_cross_encoder_reranker.py tests/test_embedder.py
```

分组避免 Windows 同时导入 Torch 造成内存压力；未设置 RUN_MODEL_TESTS。10 项跳过为已有真实模型集成测试。jieba 的 pkg_resources 弃用警告仍存在，两进程分别报告同一警告。另先行运行 Docker 合约与新增部署测试合计 11 项通过，此数量属于上述全量的子集，不重复累加。

前端 22 项 API/协议测试及 15 项组件测试覆盖：真实 multipart 字段、成功/重复/失败、拖放、普通问答、SSE 字节切分（含中文）、CRLF/多行 data、sources 先于 token、done/error、提前断流、无文本、非法 JSON、no_evidence、不可用服务及重试连接、停止与部分答案保留、来源折叠/展开、回车/换行/输入法、多轮展示与清除、手机抽屉焦点约束、外部图片禁止自动加载。

核心测试先观察失败再实现；审查发现的图片加载风险和手机焦点问题也先新增失败测试，再修复。曾发现新增测试用了 Testing Library 不支持的 `exact` 类型选项，已删除；最终 `npm run build`（包含 tsc）通过。最终 JS 365.80 kB / gzip 114.04 kB，CSS 16.00 kB / gzip 4.58 kB。npm 安装有一个 whatwg-encoding 传递依赖弃用提示。

### 容器与浏览器实测

- Node 22 / 非 root Nginx 多阶段镜像构建成功，Nginx `-t` 和 `docker compose config --quiet` 通过。一次重建遇到 Docker Registry 认证端点网络超时，继承宿主已有代理到构建子进程后成功；未更改全局代理、.env 或 Provider。未下载任何模型。
- frontend 实际以 UID 101 运行。最终 rag-api 和 frontend 均 healthy；只重建前端，原后端容器 ID 和镜像 ID 完全相同。证据见 `frontend_container_validation_2026-11-07.json`。
- 实际入口 **http://127.0.0.1:8080/**；页面、`/api/health`、`/api/ready` 均 HTTP 200，分别取得真实 `ok`/`ready`。
- 浏览器真实 PDF 重复上传在 Vite Proxy 和最终 Nginx 入口均通过：`docker-acceptance-20261107.pdf`，已有 document_id `bfa523d6243c45fc8eca7e7ea82dae4f`、1 Chunk、duplicate。没有新增/覆盖知识库文件。本轮没有通过真实服务发送聊天请求。
- 独立 5174 页面只连接本地 mock HTTP 服务 18001；浏览器实际验证了普通回答、sources 先于实时文本、done、无证据提示、中途 error 保留部分内容和来源展开。模拟资料及答案均明确写出“测试夹具/模拟”，不能作为真实检索或 DeepSeek 回答证明。
- 检查 1440×900 桌面和 390×844 手机布局，手机实际 document.scrollWidth=390，无横向溢出；知识库抽屉与回答来源可操作。最终生产页控制台 error/warn 记录为 0。
- 临时 Vite 5173/5174 和 mock 18001 进程均已关闭，生产容器继续运行。浏览器已恢复默认尺寸并保留生产入口。

截图：`frontend_production_2026-11-07.png` 为真实容器主页及重复上传；`frontend_mobile_2026-11-07.png` 为手机欢迎页；`frontend_mock_chat_2026-11-07.png`、`frontend_mobile_chat_mock_2026-11-07.png` 为**模拟回答**，不是实际模型结果。

### 数据保护与尚未验证项

修改前记录 371 个非秘密文件 SHA256；验收时 367 个逐字节不变，4 个为上述授权修改，没有删除或意外变更。保护范围包含 35 个 app 文件、10 个 data 文件（含 V1/V2/V3 和上传文件/manifest）、243 个 results 文件。Day 7/Day 8 及 V3 历史结果不变。完整逐文件哈希见 `frontend_integrity_2026-11-07.json`。

本轮没有读取 .env 或密钥，没有真实 LLM 调用，没有模型下载，也没有重新运行四方案实验。只下载了 npm 依赖和前端基础镜像。独立只读审查复核了协议、隐私和 Docker 边界，已修复其确认的问题。

限制：本轮未做真实 DeepSeek 问答验收；真实新文件入库不重复执行，仅以已有后端回归测试和真实重复上传验证连接。响应式检查使用桌面浏览器手机尺寸，不代表 Safari/实体手机全兼容。前端不新增服务端会话记忆、文档删除/完整列表、认证或公网 TLS。非空上下文不保证证据充分，继续遵守后端 no_evidence 的实际语义。

## 协议参考

SSE 的空行分帧、多行 data 和注释处理按 [MDN SSE 格式说明](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format)；使用 Fetch 处理 POST，而非只支持 GET 的原生 EventSource。Nginx 的禁用响应缓冲配置参见 [proxy_buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering)，变量代理与容器地址更新参见 [proxy_pass](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass) 和 [resolver](https://nginx.org/en/docs/http/ngx_http_core_module.html#resolver)。
