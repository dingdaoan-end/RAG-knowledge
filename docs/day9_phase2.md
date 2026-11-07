# Day 9 Phase 2：输入校验与空候选

## Pipeline 接口

原接口保持不变：

```python
answer: str = pipeline.ask(question)
text = "".join(pipeline.stream_ask(question))
```

新增兼容接口：

```python
result = pipeline.ask_with_result(question)
print(result.answer, result.status)

for chunk in pipeline.stream_ask_with_result(question):
    print(chunk.answer, chunk.status)
```

`RAGResult` 包含 `answer` 和 `status`。流式接口每项的 answer 是文本增量，不是累计答案。

- `answered`：走了 LLM 生成路径，不代表内容已通过事实核查或证据充分性评估。
- `no_evidence`：检索无候选，或者重排正常返回后无候选。直接返回固定说明，不调用 LLM。
- 空字符串、纯空白及非字符串问题抛出 `ValueError("question 不能为空")`，发生在查询改写、检索、Embedding、重排和 LLM 之前。
- 流式接口保持延迟执行；校验在首次迭代时发生。关闭迭代器会关闭上游流。
- 执行异常继续抛出，不转换为 no_evidence，不静默降级。非法输入也不是 no_evidence。

## HTTP 与 SSE

`POST /chat` 保留 answer，增加 status：

```json
{"answer": "当前知识库没有检索到可用于回答该问题的资料，因此暂时无法根据现有证据回答。你可以补充相关文档或提供更多问题背景。", "status": "no_evidence"}
```

空候选属于有效请求，HTTP 为 200；非法问题仍为 422。LLM 或重排执行失败仍为 502。

`POST /chat/stream` 继续使用 token、error、done，不新增事件类型。正常模型生成的 done payload 保持 `{}`；空候选输出固定说明后，以以下事件结束一次：

```text
event: done
data: {"status": "no_evidence"}
```

旧客户端继续拼接 token 并在 done 时停止；需要区分证据不足的新客户端读取 done 中的 status。done 表示本次响应传输完成，不一定调用过模型。错误路径没有 done。

Phase 1 的有效文本检查保留：正常生成至少有非空白文本才能开始 HTTP 200；首段前失败返回 502，开始后失败发送 error；空白文本不算有效答案，成功答案的原始空白保留。

## Day 8 QA 采集兼容

新记录新增 `pipeline_status` 和 `llm_called`，纳入采集指纹。

- 正常生成继续保存实际 Prompt、上下文和来源，llm_called=true。
- 空候选保存固定回答、no_evidence、llm_called=false、prompt=null、context="" 和空来源列表。配置的 Provider 类型仍保留，但不代表被调用。
- 人工可以评审固定回复；real_llm_quality_metrics 只汇总实际调用过真实 Provider 的已评审回答，固定回复不进入该指标。
- 旧记录没有新增字段时继续按历史采集语义读取。历史文件不迁移、不重写，旧指纹保持有效。
- 没有人工评分时仍不生成质量分数。

## 范围与运行测试

本轮只判断候选列表是否为空。不依据相关性分数推断充分性，也不判断非空文档是否能回答问题。Context Builder、检索和重排算法、Provider 的超时和重试配置不变。

在 `D:\RAGrepository` 的 PowerShell 中运行：

```powershell
$env:PYTHONUTF8 = '1'
$env:RUN_MODEL_TESTS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

新增回归使用 fake 模型和实际 TextRetriever/FAISS 或 SDK 模拟，不联网下载模型。
