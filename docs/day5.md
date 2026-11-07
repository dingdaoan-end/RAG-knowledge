# Day 5：LLM 与 HTTP 服务

## 文件职责

- `config.py`：Settings.from_env() 读取 MODEL_NAME/API_KEY/BASE_URL，兼容 OPENAI_API_KEY；API Key 不出现在配置对象的 repr 中。
- `.env.example`：列出变量，无真实密钥；需要在终端设置环境变量，此文件不会自动加载。
- `app/generation/llm.py`：BaseLLM 定义 generate/stream_generate；MockLLM 逐字符 yield；OpenAILLM 用官方 Responses API 返回完整答案或提取文本增量。
- `app/generation/__init__.py`：导出 OpenAILLM。
- `app/pipeline.py`：普通和流式路径共用检索、context、prompt 构建，stream_ask 用 yield from 转发文本。
- `app/api/routes.py`：定义 POST /chat 和 POST /chat/stream，验证请求，组织 JSON/SSE，处理异常。
- `app/api/__init__.py`：标记 API 包。
- `main.py`：启动前一次性建库，创建 LLM 和 FastAPI，用 uvicorn 启动；保留 --query 单次命令行问答。
- `requirements.txt`：固定官方 SDK、FastAPI、uvicorn 和测试 HTTP 客户端版本。新版官方 SDK 使用 HTTPX2，FastAPI 的 TestClient 使用 HTTPX，分别固定依赖。
- `tests/test_config.py`：检查环境变量、密钥别名、配置 repr。
- `tests/test_llm.py`：真实官方 SDK 加 HTTP 层模拟，验证请求和 SSE 解析，不发付费请求。
- `tests/test_api.py`：检查两条路由、输入校验、prompt 传递、增量生成和异常响应。

## LLM 为什么需要抽象接口

pipeline 只依赖 generate/stream_generate，不依赖模型名称、SDK 或请求细节。MockLLM 和 OpenAILLM 都遵守同一接口，因此替换 LLM 不用改检索和 pipeline。BaseLLM 的接口抛出 NotImplementedError，提醒子类实现；close 默认为空操作，网络实现覆盖它管理连接生命周期。

```python
def generate(self, prompt):
    response = self.client.responses.create(model=self.model, input=prompt)
    return response.output_text
```

实际实现还检查 response.status 和文本是否为空，避免把生成失败误当成功。

## API Key 为什么不能写死

Key 是账户访问凭证。写进代码会随 Git 或共享文件传播，泄露后可能被用于调用模型。环境变量将代码与每台机器的凭证分开；Settings 的 repr 隐藏密钥，接口错误也不返回 SDK 的原始错误内容。现有 .gitignore 忽略 .env，仅允许提交无密钥的示例。

MODEL_NAME 默认 gpt-4.1-mini，可设置为账户可用模型；BASE_URL 不设置时使用官方地址，自定义地址必须实现 Responses API 的 /responses 及其流式事件协议。此实现不自动把请求转换为 Chat Completions。

## Streaming 与普通返回

普通返回等待模型生成整段答案，再返回 JSON。Streaming 在生成过程中就发送增量，缩短用户看到首段内容的等待时间，不代表完整推理一定更快。

```python
with self.client.responses.create(model=self.model, input=prompt, stream=True) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
```

真实事件可能包含多个 token；MockLLM 每次 yield 一个字符。实现还检查 completed/failed/incomplete/error 事件；上游流正常结束、异常或调用方关闭生成器时都会关闭。

```python
def stream_ask(self, question):
    yield from self.llm.stream_generate(self._build_prompt(question))
```

这里不是先调用 ask 拿完整答案再切块。API 按 SSE 发送：

```text
event: token
data: {"token": "RAG"}

event: token
data: {"token": " 是检索增强生成。"}

event: done
data: {}
```

客户端依次拼接 token 字段，done 表示成功结束。多次 yield 可能被 HTTP 或客户端缓冲合并，所以客户端应解析 SSE 帧，不能假设一次网络读取对应一个模型 token。

发送首段前的异常返回 HTTP 502。开始 streaming 后响应头已经发出，无法改 HTTP 状态码，因此用 error 事件报告失败，并且不发 done。缺失、空白或非字符串 question 返回 HTTP 422。

ChatStreamingResponse 是 StreamingResponse 的一个小子类，只增加响应生命周期最外层的清理：即使客户端在发送响应头或正文时断开，也显式关闭异步响应生成器与上游 LLM 生成器。仅在正文生成器内部写 finally 不能覆盖发送失败时生成器尚未开始的情况；测试覆盖 ASGI 2.3 取消与 2.4 发送异常。

## FastAPI 在系统中的作用

FastAPI 负责 HTTP 请求入口、数据验证、调用 pipeline、JSON/SSE 返回和服务生命周期。它不承担 embedding、检索和 prompt 的业务逻辑。同步模型调用在线程池运行，避免阻塞事件循环；服务停止时关闭 SDK 连接。

```text
启动：环境配置 → Loader → Splitter → 文档 Embedding → FAISS 索引（复用）

请求：POST /chat 或 /chat/stream
      → question 验证
      → RAGPipeline
      → TextRetriever → 查询 Embedding → FAISS Top-K
      → Context Builder → Prompt Builder
      → generate / stream_generate
      → OpenAI Responses API
      → 完整 JSON / SSE 文本增量
```

接口只接收问题，服务当前加载启动参数 --file 指定的单个 txt/Markdown 文件；修改资料后重启。默认监听本机 127.0.0.1:8000，接口文档在 /docs。

## 运行与验证

```powershell
# API Key 在终端设置，配置说明见 README
.\.venv\Scripts\python.exe main.py
# 无密钥测试 API
.\.venv\Scripts\python.exe main.py --mock
# 单次命令行问答
.\.venv\Scripts\python.exe main.py --mock --query "什么是 RAG？"
# 完整测试，包含真实 BGE
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

官方依据：[OpenAI Python SDK](https://developers.openai.com/api/reference/python)、[OpenAI Streaming](https://developers.openai.com/api/docs/guides/streaming-responses)、[FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)。
