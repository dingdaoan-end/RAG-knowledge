# Phase 1：上传、双索引快照与来源引用

## 已批准范围与执行计划

本轮按用户 2026-11-06 的完整规格直接执行，不进入 Docker 阶段。

1. 先补解析、来源、入库和 HTTP 组合回归测试，确认缺失行为失败。
2. 复用已有 pypdf 解析文本型 PDF；新增 multipart 依赖。保留旧本地 TXT/Markdown loader 的契约。
3. 在独立服务中维护原始知识库 chunks、上传 manifest 和检索快照。串行上传，查询不获取写锁。FAISS/BM25 始终引用相同顺序的 Document 对象；双索引及磁盘提交成功后整体发布。
4. 原文件使用服务端 UUID 命名，内容 SHA256 去重。原始文件先暂存，manifest 用同目录临时文件和 os.replace 提交；异常清理本次文件，保留旧快照。
5. 原始文件与 manifest 是持久化真源，重启校验文件/指纹再重建双索引。单进程服务；不实现多进程写入或数据库。
6. Pipeline 一次准备上下文和 sources，不二次检索，不在实例上保存每个请求的临时来源。保留字符串接口；普通响应新增 sources，SSE 在 token 前发送 sources 事件，保持 502/error/done 规则。
7. 集成启动、可配置上传目录/大小和空知识库；执行相关及全量离线测试，复核算法、历史文件与密钥配置边界。

## 实现决策

- 用户这次明确要求 sources 默认返回，取代上一轮建议的 opt-in 方案。旧 HTTP 精确字段/事件断言仅按新契约扩充，保留原失败断言。
- 使用原仓库工作目录：已有文件均未跟踪且没有可作为基线的 Git 提交，新 worktree 会丢失当前工作内容。因此用修改前 SHA256 快照核验范围，不建立新 checkout。
- 只支持 UTF-8 文本及文本型 PDF。扫描/加密/坏 PDF 明确拒绝；不新增 OCR。
- 默认融合参数和 Mock Reranker 不变，真实精排需要显式 --reranker cross-encoder。
- 测试只替代 embedding/LLM 推理，实际使用解析器、Splitter、FAISS、BM25 和 Hybrid；不调用真实 API、不下载模型。

## 验证记录

验证日期：2026-11-06（Asia/Shanghai）。

### 修改范围

新增：

- `app/ingestion/service.py`：上传事务、稳定 ID、内容去重、双索引快照、manifest 与恢复。
- `app/api/documents.py`：multipart 上传端点和安全错误响应。
- `app/api/upload_limits.py`：multipart 解析前的总字节限制，支持 Content-Length 预检与无长度流式计数；挂载子应用也受保护。
- `tests/test_upload_parsing.py`、`tests/test_ingestion_service.py`、`tests/test_documents_api.py`、`tests/test_upload_startup.py`、`tests/test_sources.py`：118 个新增离线用例。
- 本文、`phase1_upload_before_sha256.json` 和 `phase1_upload_validation.json`：实现与完整性验证记录。

修改：

- `app/ingestion/loader.py`：保留旧 TXT/Markdown 本地接口，新增上传字节校验和 PDF 按页解析。
- `app/pipeline.py`：RAGResult 增加 sources，单次准备实际 Prompt 与来源，保持字符串接口。
- `app/api/routes.py`：普通响应 sources 与 SSE sources 事件；保留原失败/关闭协议。
- `main.py`：装配入库服务、空知识库、恢复与上传限额中间件；增加可注入 embedder 便于离线启动测试。
- `config.py`：只新增 UPLOAD_DIR/MAX_UPLOAD_BYTES，不改原模型、密钥或 BASE_URL 读取方式。
- `requirements.txt`：新增 pypdf==6.20.1、python-multipart==0.0.33；本地 pypdf 已存在，只安装了 multipart。
- `tests/test_api.py`、`tests/test_pipeline_reliability.py`：共 3 处既有响应/事件精确断言加入用户要求的新 sources 契约，原错误断言保留。
- `README.md`、`.gitignore`：运行和限制说明；忽略默认上传目录。

### 回归过程与结果

所有测试使用假 embedding/LLM 或现有 fake scorer，不调用真实 API，不下载模型。解析、FAISS、BM25L、Hybrid 使用实际代码。

先补测试确认缺失行为失败，再实现并验证。额外经过 RED → GREEN 的边界包括：

1. manifest 原子替换后不再清理已被移动的临时路径，避免清理异常触发错误回滚。
2. 相同字节换成不匹配的扩展名时仍被拒绝，去重不能绕过正文/签名验证。
3. 当前上传上限降低时，先前合法文件仍可按大小/SHA 恢复；新上传按新限额拒绝。
4. 有/无 Content-Length 的超大 multipart 均提前停止读取，并关闭解析临时文件。
5. 应用挂载到 `/rag` 时仍执行同样的请求大小限制。

最终全量测试分成互不重叠的两组串行执行：

| 测试组 | 通过 | 失败 | 跳过 | 耗时 |
|---|---:|---:|---:|---:|
| 除下述两个测试文件外的全量测试 | 565 | 0 | 0 | 32.96 s |
| test_cross_encoder_reranker.py + test_embedder.py | 38 | 0 | 10 | 53.00 s |
| 总计 | 603 | 0 | 10 | — |

原有 485 个非模型测试继续通过，新增 118 个测试通过。10 个跳过项是 RUN_MODEL_TESTS 未启用的真实 BGE 集成测试。各组都有同一条既有 jieba/pkg_resources 弃用警告。

可复现命令（PowerShell，项目根目录）：

```powershell
cd D:\RAGrepository
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --ignore=tests/test_cross_encoder_reranker.py --ignore=tests/test_embedder.py --tb=short
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_cross_encoder_reranker.py tests/test_embedder.py --tb=short
```

执行时先检查 RUN_MODEL_TESTS 未设置为 1，避免违反本轮离线约束。PYTHONUTF8 只统一父子测试进程编码，不改变 Provider 配置。

早期合并执行曾在 torch 导入/异常格式化时出现原生进程中止或 MemoryError，未算作通过。改为串行分组后跑通。一次父进程使用 `-X utf8`、子进程默认编码的运行触发 CLI 帮助输出解码失败；向子进程继承 PYTHONUTF8 后原测试原样通过，没有修改该测试预期。

### 本地 HTTP 验收

通过临时目录和真实 Uvicorn 监听本机回环地址执行一次服务启动、停止和重启；Embedding 为 TinyEmbedder，LLM 为读取实际 Prompt 证据的 fake，不属于真实模型质量验收。

| 操作 | 实际结果 |
|---|---|
| 空知识库 `/chat` | 200 / no_evidence，LLM 调用 0 次 |
| 上传 smoke.txt | 201 / indexed |
| 不重启立即 `/chat` | 200 / answered；答案包含上传正文，来源文件 smoke.txt、chunk_id=0 |
| `/chat/stream` | sources → token → token → done，无 error |
| 相同字节以 renamed.md 再上传 | 200 / duplicate，相同 document_id |
| 重启后再问 | 200，正文与稳定 document_id 保持 |

TXT、Markdown、文本 PDF 的真实解析上传、无正文 PDF、坏签名等由对应自动化测试补充覆盖。本地临时服务已关闭，测试上传目录已清理。

### 双索引与磁盘提交边界

上传写锁内先解析并构建完整新快照；FAISS/BM25 引用同序同对象的 Document，且 chunk_uid 唯一。只有双索引均成功、原始文件与 manifest 提交成功后才发布 pipeline.retriever；正在查询的请求继续使用原快照。manifest 用同目录暂存、flush/fsync 和 os.replace 写入。可捕获失败清理本次文件，旧 manifest 与快照保留。

来源只由最终 Top-K（含重排后的顺序）生成，和本次 Context Builder 输入一致；不二次检索。source 显示文件名；PDF 页码保留。当前没有额外 token 预算裁剪器，因此上下文数量预算为 top_k。

### 保护与限制

初始 332 个文件的 SHA256 已保存于 `phase1_upload_before_sha256.json`，后续逐一比较；全部 251 个 data/results 原文件不变，包括 V1/V2/V3、Day 7/Day 8 和 V3 历史采集/审核。检索、Embedding、Reranker、Context Builder、Prompt、LLM Provider、指标实现的哈希均保持不变。Git 中原有文件均未跟踪，故采用此快照作为等效差异审查依据。

仅支持单进程写入；不要让多个 workers 或多个独立服务共享上传目录。当前每次上传重新编码全部语料；大文件 PDF 的解析 CPU/解压内存尚无独立进程配额。没有 OCR、文档删除/列表 API、额外 token 预算或索引增量持久化。硬中断可留下未提交的孤立文件，恢复不会索引这些文件，尚无自动清理任务；不承诺跨文件断电事务。恢复要求 splitter/size/overlap 与 manifest 一致。

默认 MockReranker 未改动，演示真实精排需显式 `--reranker cross-encoder` 并具备本地模型。CLI `--query` 保留只查询 `--file` 的旧行为，上传恢复服务用于 HTTP 启动路径。非空上下文仍不能保证证据充分或回答正确。

本轮没有真实 LLM 调用、模型下载、四方案评估、密钥读取、生产算法修改或 Docker 开发。
