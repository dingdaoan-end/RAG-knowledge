# Day 9 Phase 3：日志与请求追踪

## 请求 ID

每个 HTTP 请求在入口生成新的 UUID4，使用 32 位小写十六进制表示。服务端忽略调用方传入的 X-Request-ID，因此不回显外部文本、不接受跨请求复用，也不把该 ID 当认证凭据。

所有响应通过 `X-Request-ID` 返回 ID，包括普通问答、SSE、422、404、502 以及框架的通用 500 响应。500 正文仍是通用错误，框架仍向服务器传播原异常。

纯 ASGI 中间件保持请求上下文直到响应发送及资源清理结束。ContextVar 为每个请求持有独立的 RequestTrace；FastAPI/AnyIO 线程池复制上下文，因此普通生成、SSE 首段预取和后续迭代使用相同 ID。请求退出时在原 ASGI 任务中复位上下文，不在可能跨线程恢复的生成器内操作 ContextVar token。

## 日志输出与字段

复用 Python 标准库 logging，不增加依赖。`create_app()` 幂等配置独立的 `rag` logger，按一行一个 JSON 输出至 stderr，不配置文件、不覆盖评估结果、不修改 root、Uvicorn 或 SDK 的日志级别。

| 字段 | 含义 |
|---|---|
| timestamp | UTC 时间 |
| request_id | 服务端生成的 HTTP 请求 ID；直接从 CLI/评估调用 Pipeline 时为 null |
| event | stage_started、stage_completed、stage_failed、stage_cancelled |
| stage | http、pipeline、retrieval、reranker、context、llm、response |
| status | started、ok、no_evidence、failed、cancelled、rejected |
| duration_ms | 有完整阶段边界时记录，用 perf_counter 计时；阶段开始事件不含耗时 |
| exception_type | 失败/取消时的异常类名，严格限制 ASCII 字符及长度，不记录异常消息 |
| http_status | HTTP 响应状态，可用时记录 |

同一异常可以同时产生底层阶段、Pipeline 和 HTTP 边界的日志。这些是同一次失败向外传播的记录，不是重试次数。

## 阶段和耗时边界

- http：进入请求追踪中间件至 ASGI 调用完成，包含 SSE 正文发送和生成器清理；不是仅测响应头返回。
- pipeline：普通问答全流程，或流式迭代至结束/关闭。
- retrieval：现有 retriever.search 调用；包括其内部 Embedding/Hybrid/BM25 工作，本轮不拆分或猜测这些子阶段的耗时。
- reranker：现有重排调用和候选合法性检查；延迟加载与评分都包含在内，未进一步拆分计时。
- context：现有 Context Builder 与 Prompt Builder 调用，不记录构造出的内容。
- llm：现有 generate 调用，或 stream_generate 的创建、消费和清理。
- response：路由检测到的失败边界，例如空生成流。没有独立开始点，因此只记录失败和类型，不编造 duration_ms。

一个边界限制：框架外层生成的未知异常 500 响应，http 失败耗时只测到原异常离开本中间件，不包含随后由外层框架发送通用 500 正文的时间；ID 由错误处理器从 scope 补回。业务已处理的 502 以及 SSE 响应不受此限制。

流式 Pipeline/LLM 的耗时含消费间隔和客户端背压，不等于模型纯推理时间。尚未提供首 token 耗时、逐批推理耗时或分布式 tracing。

## 状态与错误

- 正常阶段返回记录 ok；不代表答案已被事实核查。
- 无候选时 Pipeline 与 HTTP 记录 no_evidence，不产生 LLM 开始或成功日志，保留 Phase 2 固定回复和 SSE 状态。
- LLM 超时、重排加载/评分失败记录实际异常类型，继续原有显式失败策略，不降级、不吞业务异常。
- SSE 首段前失败仍为 502；开始后失败仍为 HTTP 200 加 error 事件，且无 done。HTTP 终止日志同时记录 http_status=200、status=failed，避免误判为成功问答。
- 空答案沿用 Phase 1 检查；实际 Provider 抛错归于 llm 阶段，路由发现无效流归于 response 阶段。
- 生成器关闭、任务取消、可检测的客户端断开记录 cancelled，并释放上游资源；客户端传输失败和模型故障不会被混为正常完成。

## 隐私与日志故障

应用日志只接受代码内控制的元数据。不记录请求体、问题、文档正文/来源、Prompt、模型输出、Authorization、API Key、异常消息、异常请求对象或内部堆栈。不读取 .env 或秘密。

不伪造上游请求 ID：本版只记录自己的 request_id，没有声称已接入 Provider 的请求 ID。

日志调用捕获日志系统自身的异常；应用 StreamHandler 的 handleError 禁止标准 logging 将 formatter/sink 错误及堆栈旁路打印到 stderr。日志输出失败会丢失观测数据，但不会更改业务答案、状态或原业务异常。

这些保护针对 rag 应用日志。第三方、服务器访问日志、外部代理和部署平台有独立配置；例如完整 URL 可能携带隐私，服务器对未知未处理异常的默认堆栈日志也需在部署侧另行审查。本轮不宣称控制了所有外部日志来源。

## 验证

在 PowerShell 中运行新增测试：

```powershell
cd D:\RAGrepository
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -q tests/test_observability.py
```

新增测试用 fake 模型、受控异常及实际 FastAPI/ASGI 请求验证关联、并发、状态、SSE 协议、脱敏、日志故障、取消和响应发送边界，不调用真实 API 或下载模型。

不修改检索/重排算法、Phase 1/2 业务语义、Day 7/8 冻结数据和历史结果。非空上下文的证据充分性、日志持久化/轮转/告警、指标平台和重试策略仍未实现。
