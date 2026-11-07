# Day 8：可复核的端到端 QA 评估

## 范围与当前结果

本版只建立数据、实际回答采集、证据保存和人工评审工作流，不自动评分。没有修改生产 Pipeline、检索器、重排器、Document 或 Day 7 冻结文件，没有新增依赖。

QA 数据版本为 1.0，共 5 题：3 个可回答问题、2 个证据不足问题。参考答案按既有 18 个评估 Chunk 编写，尚需用户复核，不根据输出调整。

本环境没有配置 API Key。实际运行使用真实 BGE Embedding + FAISS + BM25L + Hybrid + Cross-Encoder，但生成端为 **MockLLM**。5 个答案均为现有 Mock 的固定测试文本。这证明采集链路运行成功，不证明真实生成质量、拒答质量或事实正确性。

结果在 results/day8_qa_mock.json，控制台输出在 results/day8_qa_collect.txt，汇总在 results/day8_qa_summary.json。所有记录尚待人工评审，汇总显示“尚无已审核评分”，质量指标为 null。

## 数据格式与身份映射

data/evaluation/qa_eval.json 顶层有 version、corpus_sha256、metadata 和 samples。corpus_sha256 绑定 Day 7 冻结语料，不会另建或猜测文档 ID。

每个样本必须包含：

| 字段 | 含义 |
|---|---|
| id | 样本唯一标识 |
| question | 原始用户问题 |
| reference_answer | 供人工核对的参考答案，不进入 Prompt |
| required_facts | 非空且不重复的关键事实清单 |
| relevant_document_ids | 已存在的稳定 eval_id；不能写未知 ID |
| answerable | 资料是否足以支持所请求的事实或确定结论 |

answerable=true 至少关联一份现有证据。false 可以关联支持“无法确定”的资料，也可以没有相关文档。qa_004 对应示例 CUDA 编号不能证明硬件损坏；qa_005 没有当前机器的实际型号和版本信息。这两个问题不能据教学技术文档给出确定的实际环境结论。

标签不改变检索行为：即使 answerable=false，仍保存系统实际召回的上下文，不强行置空，不注入参考拒答，也不伪装已有证据足够。

## 怎样取得实际上下文

现有 RAGPipeline.ask 只返回答案字符串。本模块使用独立包装层：

1. _TraceRetriever 记录实际检索问题和返回结果，校验 eval_id 与文本、来源一致。
2. _TraceReranker 记录实际重排结果，兼容旧 Document 接口和新结果对象接口。
3. _TraceLLM 保存 generate 实际收到的完整 Prompt，再原样调用原 LLM，只调用一次。
4. 新建相同配置的 RAGPipeline，复用原组件，不修改生产实例或组件引用。
5. 根据实际返回结果与最终 Top-K 形成文档快照，调用现有 Context Builder/Prompt Builder，要求重建的完整 Prompt 与捕获的 Prompt 严格一致。无法验证时明确报错，不猜测来源。

记录包括问题、原始答案、完整 Prompt、实际 context、上下文文档 ID/正文/metadata/rank/score、检索问题、生成模式、模型配置、语料与 QA 数据指纹、UTC 时间戳和运行耗时。运行耗时覆盖问答调用，不包含脚本此前的模型初始化时间。

generation.mode 根据实际 LLM 类型记录为 mock、real_api 或 custom_unverified。给 Mock 对象填写一个真实模型名称也不会把它变成真实模型评估。配置仅采集白名单字段，不保存 API Key、SDK client 或环境变量。

## 人工评分标准

三个分数分别取 0、1、2，不相加，也不自动合成总分：

| 维度 | 0：不合格 | 1：部分合格 | 2：合格 |
|---|---|---|---|
| faithfulness | 有明显无依据断言，或与实际上下文矛盾 | 部分事实有支撑，但部分断言或限制交代不足 | 事实性断言均由实际上下文支持，证据限制表述恰当 |
| answer_relevance | 没回应问题，或只有无关内容 | 仅回应部分问题，含明显偏题内容 | 直接回应核心问题；证据不足时明确说明不能确定及缺少什么 |
| factual_correctness | 关键事实或结论错误、严重误导 | 局部事实有误，但主要事实基本正确 | 核心事实正确，无实质性事实错误 |

Faithfulness 只对照实际 context。某句话出现在 context 中，不代表现实中一定正确。Factual Correctness 需要核对参考答案或可靠外部依据，并在 factual_correctness_basis 记录依据及版本/出处。参考答案本身也可能需要纠正，不能视作无条件真理。

答案缺少事实属于覆盖问题，不应简单混同为事实错误。没有可核查陈述或无法判断时，可以保持该分数 null、状态 in_progress，并写明原因，不强行完成评审。

### 关键事实覆盖

required_facts_covered 必须与 required_facts 逐项同序对应：

```json
{
  "fact": "不能保证完全消除幻觉",
  "covered": null,
  "answer_quote": null,
  "notes": ""
}
```

人工把 covered 改为 true 或 false。true 必须摘录实际答案中支持该判断的原文；引文需要确实出现在答案中。false 可以不写引文，并说明缺失之处。不根据长度、关键词命中或参考答案的字面重合自动计分。

### 无依据断言

初始 unsupported_claims=null 表示尚未检查。完成检查且没有发现时写 []；有问题时逐条填写：

```json
{
  "claim": "待检查的事实性断言",
  "answer_quote": "答案中真实存在的原文片段",
  "reason": "为什么这些上下文不能支持该断言",
  "context_document_ids": []
}
```

可关联实际上下文文档 ID，不能关联没有提供给模型的文档。无依据不等于现实中必然错误，两种判断需要分开。

### 证据不足样本

对于 answerable=false，人工填写 insufficient_evidence_handled：true 表示明确说明资料不足、避免无依据确定结论，并适当说明需要的补充信息；false 表示未妥善处理。初始 null，系统不自动从措辞判断。

## 人工评审操作

先保留原始采集文件，再复制一个评审工作文件：

```powershell
cd D:\RAGrepository
Copy-Item -LiteralPath results/day8_qa_mock.json -Destination results/day8_qa_reviewed.json
```

只编辑每条记录的 review：

1. 阅读 question、answer、实际 context 和文档快照，再看参考答案与 required_facts。
2. 独立填写三个维度，逐事实判断覆盖，检查 unsupported_claims。
3. 对负例填写 insufficient_evidence_handled。
4. 写 reviewer、含时区的 reviewed_at、factual_correctness_basis 和 review_notes。
5. 未完成时用 pending/in_progress；所有必填项完成后，由人工改为 completed。

例如 reviewed_at 使用 2026-10-20T12:00:00+08:00 这样的格式。程序检查格式和引文可追溯性，不验证评审者身份，也不会代替人工判断语义。

capture_sha256 覆盖 review 以外的采集内容。误改答案、上下文、参考标签或模型信息会在加载时触发错误；只编辑 review 不影响指纹。它用于发现误操作，不是防篡改签名或身份认证。

汇总评审：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_qa.py summarize --input results/day8_qa_reviewed.json --output results/day8_qa_reviewed_summary.json
```

只对 completed 记录计算各维度均值。关键事实覆盖率是已完成评审中的已覆盖事实总数/事实总数；无依据断言计数单独报告；证据不足处理率只对已完成的负例计算。未完成记录不当作 0 分。

Mock 或 custom_unverified 记录即使经过人工审核，也只汇总为对应材料的 human_review_metrics；real_llm_quality_metrics 保持 null，不能冒充真实 API 模型质量。若没有任何 completed，显示“尚无已审核评分”，所有质量统计均为 null。

## 运行采集

```powershell
# 已缓存真实检索/重排模型，生成端明确选择 Mock。
.\.venv\Scripts\python.exe scripts/evaluate_qa.py collect --llm mock --local-files-only --output results/day8_qa_mock_v2.json

# 配好现有 API_KEY/OPENAI_API_KEY、MODEL_NAME、BASE_URL 后，显式使用真实 LLM。
# 不配置密钥时会报错，不自动替换为 Mock。
.\.venv\Scripts\python.exe scripts/evaluate_qa.py collect --llm real --local-files-only --output results/day8_qa_real.json

# 汇总不需要模型、密钥或网络。
.\.venv\Scripts\python.exe scripts/evaluate_qa.py summarize --input results/day8_qa_mock.json --output results/day8_qa_summary_v2.json

# 新测试不下载模型。
.\.venv\Scripts\python.exe -m pytest -q tests/test_qa_evaluation.py
```

local-files-only 仅限制本地检索/重排模型加载；若选择 real，生成 API 仍需要联网。所有输出使用新文件名，拒绝覆盖已存在的结果、评审或冻结文件。

## 当前实际样例

qa_001 的实际问题：RAG 为什么可以降低大模型幻觉？它能保证完全消除幻觉吗？

实际来源为 rag_evidence、rag_limits、rag_terms。MockLLM 实际答案为“这是基于知识库资料生成的测试回答”。忠实度、切题性、事实正确性及逐事实覆盖均未由系统打分，review.status=pending。

qa_005 的相关证据列表为空，但实际检索仍返回 cuda_oom、cuda_1175、rag_overview。程序保留这些真实上下文，不能把技术文档当成本机环境的事实。其限制处理仍待人工判断，不能据 Mock 输出认定系统通过了拒答测试。

## 当前限制

只有 5 个教学样本，参考答案和标注尚待人工复核；没有真实 LLM 输出，尚不能得出生成质量结论。实际 Prompt 格式若变化，包装层会报错，需要适配后再采集。批次遇到调用失败直接报错，不跳过问题或伪造答案。本版不提供自动评分、评审者认证或多评审者一致性计算。

完成到此停止，不开展 Day 9。
