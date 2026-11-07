# 第一版 Cross-Encoder Reranker

## 范围

只增加 Hybrid 候选之后的重排。FAISS IndexFlatIP、BGE Embedding、BM25L、Min-Max、alpha=0.7、候选并集和 Hybrid final_score 均保持不变。

```text
知识库 -> Vector + BM25 -> Hybrid Top-N
                              |
                       (query, candidate.text)
                              |
                         Cross-Encoder
                              |
                    按 rerank_score 降序
                              |
                         Top-K -> Context -> LLM
```

Reranker 不接收完整知识库，不持有 FAISS/BM25 索引，只处理调用者传入的候选。因此即使知识库有十万条，Hybrid 只返回二十条时，也只构造二十个文本对。

## 接口和结果

```python
from app.retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-base",
    batch_size=16,
    max_length=512,
)
candidates = hybrid.search(question, top_k=20)
results = reranker.rerank(question, candidates, top_k=3)
```

`RerankResult` 保存：

- `document`：原 Document 对象，metadata 不变。
- `rerank_score`：Cross-Encoder 输出的原始 logit，可为负数，不是概率。
- `rank`：重排后的名次，从 1 开始。
- `vector_score`、`bm25_score`：原始单路分数。
- `normalized_vector_score`、`normalized_bm25_score`：原归一化分数。
- `final_score`：原 Hybrid 融合分数；`hybrid_score` 是它的读取别名。
- `retrieval_score`、`retrieval_rank`：原候选的 score/rank，方便回看重排前状态。
- `score` 属性：指向 rerank_score，使现有 Context Builder 直接使用重排后的分数和名次。

也可以接收已有 SearchResult；它没有的 Hybrid 字段保持 None。不会推测单路分数的类型或重新构造检索分数。

## 核心代码如何工作

1. 先检查 query、top_k 和重复候选，空候选直接返回空列表。
2. `pairs = [(query, result.document.text) for result in hybrid_results]`：每个候选与相同问题组成一对。
3. `CrossEncoder.predict(pairs, batch_size=...)`：联合编码每对文本；batch_size 控制每批推理数量。
4. 验证每个候选只有一个有限实数分数。NaN、正负无穷、复数、错误数量或多分类形状报 ValueError。
5. 稳定降序排序候选索引，同分保持 Hybrid 顺序，然后截取 top_k。
6. 创建新的 RerankResult，重新编号 rank，复制检索分数字段，保留原 Document 引用；不修改输入结果。

没有将 Cross-Encoder 分数与 Hybrid 分数相加，也没有对 Cross-Encoder 分数做 Min-Max。

## 模型及为什么不能预计算固定分数

默认模型是 [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)，支持中文和英文，可使用现有 sentence-transformers 的 CrossEncoder；没有增加依赖。当前机器只有 CPU，模型权重约 1.11 GB，因此使用小批候选做学习实验。

Embedding 模型可以单独编码文档。Cross-Encoder 则把 query 与 document 一起输入，文本之间通过注意力交互：同一段文字面对两个不同问题，会有不同的相关性分数。因此不能离线为文档保存一个适用于所有问题的固定 rerank_score。

明确配置 Identity 激活，保留模型原始 logits。模型只在首次处理非空候选时加载，后续复用；加载失败正常报错，不会静默退回预设分数。

## 与旧代码的兼容

原 Reranker/MockReranker 及旧扩展接收 Document 列表，继续保留。CrossEncoderReranker 声明 `accepts_results=True`，Pipeline 将完整候选结果传入；旧实现仍收到 Document 列表。这个小分支用于保留原接口，无需重写旧测试和 Query Rewrite。

Pipeline 面向用户原问题重排；重排完仍检查返回的文档来自候选且不重复。Context Builder 通过结果对象的 score 属性工作，无需改写。

## 边界规则

| 输入 | 行为 |
|---|---|
| 空白或非字符串 query | ValueError |
| 空候选 | []，不加载模型 |
| top_k 非正整数，或 bool | ValueError |
| top_k 大于候选数 | 返回全部候选 |
| 单个候选 | 仍进行评分，rank=1 |
| 同一 Document 对象重复出现 | ValueError，避免重复占用上下文 |
| 文字相同但对象不同 | 保留为不同候选，沿用当前对象身份约定 |
| 分数相同 | 保留原候选顺序 |
| 异常分数 | ValueError，不跳过或伪造分数 |

## 运行

在 PowerShell 中：

```powershell
cd D:\RAGrepository

# 真实 BGE 检索 + 预设重排分数，仅用于观察排序和 Top-K。
.\.venv\Scripts\python.exe demo_reranker.py --fake

# 真实 BGE + 真实 Cross-Encoder；模型首次使用可能下载。
.\.venv\Scripts\python.exe demo_reranker.py

# 已缓存时，只从本地加载重排模型。
.\.venv\Scripts\python.exe demo_reranker.py --local-files-only

# 接入已有完整问答流程；MockLLM 不调用云端。
.\.venv\Scripts\python.exe main.py --mock --retrieval hybrid --candidate-k 10 --top-k 3 --reranker cross-encoder --reranker-model BAAI/bge-reranker-base --query 'RAG 是什么？'

# 新增单元测试：不需要下载重排模型。
.\.venv\Scripts\python.exe -m pytest -q tests/test_cross_encoder_reranker.py

# 原有真实 BGE 测试在本地模型已缓存时一并运行。
$env:RUN_MODEL_TESTS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

Demo 的 5 篇短文包含术语索引、解释机制的正文和无关技术文档，Hybrid 只给 Reranker 四条候选。fake 模式按候选位置预设 0.62、0.95、0.71、0.10，仅说明排序过程，不代表语言理解能力；真实模式独立输出模型分数。

## 本次实际验证

完整测试：198 项通过，包含原有 160 项和新增 38 项；原测试文件未修改，无跳过。使用 fake scorer 的新增测试验证候选范围、排序、Top-K、字段保留、异常分数、模型配置、加载失败、命令行和普通/流式 Pipeline，以及未继承基类的旧接口实现。真实模型不作为单元测试的下载前置条件。

真实 BGE + Cross-Encoder 已在本地 CPU 上运行成功。完整输出见 [reranker_demo_real.txt](reranker_demo_real.txt)，预设分数的输出见 [reranker_demo_fake.txt](reranker_demo_fake.txt)。

问题：RAG 为什么可以降低大模型幻觉？

| 文档 | Hybrid 排名 | Hybrid 分数 | 重排名次 | rerank_score |
|---|---:|---:|---:|---:|
| C：解释外部知识减少参数知识依赖 | 2 | 0.827088 | 1 | 8.835895 |
| A：关键词较多的术语索引 | 1 | 0.955508 | 2 | 2.350604 |
| B：解释将外部证据放入提示词 | 4 | 0.034517 | 3 | -5.054113 |

Hybrid 候选顺序为 A > C > E > B；重排 Top-3 为 C > A > B，CUDA 文档 E 被移出最终结果。正文 C 超过了术语索引 A，但 A 仍排在正文 B 前，说明该样例也不能证明模型总能排出理想顺序。负分是合法 logit，Top-K 也可能包含低相关文档；本版没有增加阈值过滤。

保护文件的 SHA256 校验确认 hybrid.py、retriever.py、embedder.py、Document 定义与 requirements.txt 均未修改。

## 成本和当前限制

设候选数为 N，问题与文档合计 token 数为 L，模型结构固定：标准 Transformer 的注意力计算随 L 约为平方增长，候选推理成本约 O(N L²)，排序 O(N log N)。这不包含前一阶段的检索成本。batch_size 减少调用开销、控制内存，不会减少需要评分的候选数。

最大长度限制针对问题与文档组成的文本对，超长输入会被截断；重要信息可能因此丢失。CPU 重排增加延迟。Reranker 只能调整已召回候选的顺序，无法找回 Hybrid 没有召回的资料。有限 Demo 不能证明在所有问题上均有提升。
