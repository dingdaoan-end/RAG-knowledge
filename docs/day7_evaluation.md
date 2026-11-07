# Day 7：检索评估基线

## 本次范围与文件

只新增评估模块、数据、测试、独立入口和结果文件。原有 Retriever、Cross-Encoder、Pipeline、Context Builder、Document、依赖及旧 Recall 脚本不修改。

| 文件 | 职责 |
|---|---|
| app/evaluation/dataset.py | 验证数据集、建立独立稳定 eval_id、计算数据指纹 |
| app/evaluation/retrieval_metrics.py | 纯函数实现 Recall、RR/MRR、DCG/nDCG |
| app/evaluation/evaluator.py | 四方案统一调用、候选校验、宏平均、逐问题报告和摘要 |
| scripts/evaluate_retrieval.py | 加载现有模型和检索器，运行并保存真实实验 |
| data/evaluation/retrieval_eval.json | 18 个 Chunk、14 个问题及预先编写的分级标注 |
| tests/test_retrieval_metrics.py | 指标手算及边界测试 |
| tests/test_retrieval_evaluator.py | 标识映射、四方案、候选范围、JSON 和入口测试 |
| results/day7_retrieval.json | 机器可读结果，包含逐 Query 排名、分数及实际候选数量 |
| results/day7_retrieval.txt | 完整控制台输出 |

## 数据与标识

JSON 顶层是 metadata、documents、queries。每条 documents 已是一个检索单元，不再切块：

```json
{
  "id": "rag_evidence",
  "source": "rag.md",
  "chunk_id": 1,
  "text": "RAG 可以降低大模型幻觉，因为它向模型提供可参考的外部事实和证据……"
}
```

评估 Loader 把 id 写入 Document.metadata["eval_id"]。它是数据集内全局唯一的持久标识，与 Python 对象的内存地址无关；原 Document 类不变。source/chunk_id 组合也必须唯一，不同源文件的 chunk_id 可以重复。结果映射时检查 ID 存在、无重复，并核对文本和来源，无法匹配即报错。

```json
{
  "id": "q01",
  "query": "RAG 为什么可以降低大模型幻觉？",
  "relevant_documents": [
    {"id": "rag_evidence", "relevance": 3},
    {"id": "rag_overview", "relevance": 2},
    {"id": "rag_limits", "relevance": 1},
    {"id": "rag_terms", "relevance": 0}
  ]
}
```

等级统一为：0 不相关；1 有部分帮助；2 相关、回答重要部分；3 高度相关、直接回答核心问题。未列出的文档按 0 处理。因此需要复核整份语料，不能仅标一个已知答案后就默认其余一定无关。

本数据集是按文意预先编写的**教学标注初稿，尚未由用户人工复核**，不是经过独立多人标注的基准。没有使用模型评审，也没有根据实验分数修改 Ground Truth。CUDA 1175/1176 只是示例日志标识，不是官方错误码知识库。

第一次模型运行前的数据集 SHA256：

```text
085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71
```

## 三个指标如何计算

所有指标采用从 1 开始的排名，跨问题取算术平均（macro mean），每个问题权重相同。

**Recall@K**：把等级 1、2、3 都作为相关。Top-K 命中数除以该问题的全部正相关标注数。例如三个相关 Chunk 只召回两个，Recall=2/3。分母来自同一份 Ground Truth，不是候选列表中已有的相关文档数。

**RR/MRR**：首个相关结果位于 r，则 RR=1/r；没有命中则为 0。对各问题 RR 求平均得到 MRR。first_relevant_rank 也单独记录，未命中为 null。

本实验统一最多获取 10 个候选，故明确报告 **MRR@10**，不是不截断的全库 MRR。纯函数 reciprocal_rank 对传入完整列表计算；若需更大范围，改变所有方案共同的 candidate-k 后重新实验。不能把只看 Top-3 的重排结果与 Top-10 的其他方法比较 MRR。

**nDCG@K**：

```text
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)，i 从 1 开始
IDCG@K = 将全部标注等级降序排列后的理想 DCG
nDCG@K = DCG@K / IDCG@K
```

等级 3、2、1 对应增益 7、3、1。手算示例：[3,2,1] 的 DCG@3 = 7 + 3/log2(3) + 1/2。nDCG 不仅看有没有命中，也看高等级资料是否排在前面。没有相关命中时为 0；理想排序为 1。

无正相关 Ground Truth 的问题直接报 ValueError，不静默计 0，也不自动排除。重复/无效/未知标识和非法等级也报错。

## 四种方案的公平调用

每个问题依次调用现有 Vector、BM25、Hybrid.search(query, top_k=10)。Vector 和 Hybrid 实际返回 10 条；BM25 只返回有正词项分数的结果，实际可能少于 10 条，不补齐。

Hybrid 的同一份实际候选交给 reranker.rerank(query, candidates, top_k=len(candidates))。返回完整候选排序，随后才对各 K 计算指标。评估器检查重排前后的 ID 集合完全相同：不能新增候选，也不能提前只保留三条。没有候选时不调用重排模型。

候选不足 K 时，使用实际结果计算，分母仍保留全部 Ground Truth。默认 18 条语料只给 Cross-Encoder 10 条 Hybrid 候选，没有直接扫描全库。

## 如何运行和复现

在 PowerShell 中：

```powershell
cd D:\RAGrepository

# 模型已缓存时完全离线运行四方案。
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --local-files-only

# 自定义数据和输出路径；首次下载模型时可省略 --local-files-only。
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --dataset data/evaluation/retrieval_eval.json --output results/day7_retrieval.json --ks 3 5 10 --candidate-k 10 --local-files-only

# 只运行新评估测试，不加载真实模型。
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_metrics.py tests/test_retrieval_evaluator.py

# 包含原有真实 BGE 测试的完整套件。
$env:RUN_MODEL_TESTS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

输出默认写到项目 results/day7_retrieval.json。记录数据集 SHA256、问题与文档数量、K、共同候选预算、alpha、模型名称、batch size、重排最大长度及 Python/库版本。查询结果包含有序候选、相关性等级、分数调试字段、首个相关名次、RR 和各 K 指标。

根目录 evaluate_retrieval.py 保留原来 Recall-only 的行为；新的四方案入口位于 scripts/，两个脚本不混用。

## 测试与保护检查

新增评估测试 57 项通过；全量 255 项通过、0 失败、0 跳过，原有 198 项测试文件未改动并继续通过。唯一警告是原 jieba 上游的 pkg_resources 弃用提示。新增单元测试不需要联网或下载模型，真实四方案实验另行运行。

对原 app/ 和 tests/ 文件的 SHA256 检查全部一致；冻结数据集指纹也一致。入口拒绝用输出文件覆盖 Ground Truth，Loader 拒绝 JSON 重复字段和与固定指标规则冲突的元数据，避免静默改变实验含义。

## 实际总体结果

模型：BAAI/bge-small-zh-v1.5 与 BAAI/bge-reranker-base；alpha=0.7；候选预算 10。

| Method | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@3 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Vector | 0.880952 | 0.940476 | 0.976190 | 0.964286 | 0.944435 | 0.958292 |
| BM25 | 0.833333 | 0.976190 | 1.000000 | 0.892857 | 0.863240 | 0.908486 |
| Hybrid | 0.952381 | 0.976190 | 0.976190 | 0.928571 | 0.930994 | 0.940820 |
| Hybrid + Reranker | 0.952381 | 0.976190 | 0.976190 | 1.000000 | 0.983605 | 0.993430 |

实际平均候选数：Vector 10、BM25 9、Hybrid 10、Hybrid + Reranker 10。完整 JSON 中保留了每个问题的实际数量。

## 逐问题示例与结论

q04：示例日志里 CUDA error 1175 应该根据什么线索排查？Ground Truth 只有 cuda_1175 为 3，cuda_1176 和 cuda_oom 为 0。

| Method | Top-3（括号内为相关性） | Recall@3 | RR@10 | nDCG@3 |
|---|---|---:|---:|---:|
| Vector | 1176(0), 1175(3), oom(0) | 1 | 0.5 | 0.630930 |
| BM25 | 1176(0), 1175(3), oom(0) | 1 | 0.5 | 0.630930 |
| Hybrid | 1176(0), 1175(3), oom(0) | 1 | 0.5 | 0.630930 |
| Hybrid + Reranker | 1175(3), oom(0), 1176(0) | 1 | 1 | 1 |

1176 的正文也提到 1175，因此关键词命中并不代表直接回答了用户的问题；BM25 本次也排错了第一名。重排改变的是顺序，Recall@3 在这个例子没有变化，RR 和 nDCG 提高。

q01 也暴露了融合的不足：Vector 把 rag_evidence 排第一，Hybrid 被关键词较多的 rag_terms 干扰，rag_evidence 降到第二。其 nDCG@3 从 Vector 的 0.798485 降为 Hybrid 的 0.523434，重排后为 0.812424。不能仅凭采用 Hybrid 就宣称更好。

q02 有三个正相关标注：faiss_ip、cosine_unit 和弱相关的 embedding_bge。BM25 Top-10 包含三者，而 Hybrid Top-10 缺少 embedding_bge；重排也无法找回。因此 Hybrid 和重排的 Recall@10 都为 2/3。这也是教学标注中“部分背景是否应算相关”需要用户独立复核的例子；本次没有为改善分数而修改该标注。

总体上，Hybrid 改善了靠前结果的覆盖，但 MRR@10 和 nDCG@3 均低于 Vector，说明融合可能干扰排序。Reranker 在相同候选集内改善了 MRR@10 和 nDCG@3，但 Recall@3 未变。BM25 的 Recall@10 最高，却不代表首位排序最好。所有方法在本数据集上都至少找到一个相关结果，没有完全未命中的问题。

## 扩展数据和下一步建议

新增问题时先查看整份语料、确定正相关及等级，再写入 queries；新 id 不得重复。新增 Chunk 时填写全局唯一 id 和唯一 source/chunk_id，保留已存在 ID 的含义。重新运行将得到新的数据指纹，旧报告不要覆盖为不同版本而不留记录。

最值得先做的是人工复核并扩展真实用户问题，尤其是近似错误编号、同义问法、否定表达、弱相关背景及无答案问题；无正相关问题需要单独制定评估规则，当前入口会明确拒绝，不应硬塞成零分。

这只是 18 篇短文、14 个问题的教学基线，没有独立调参/测试划分，也没有统计显著性结论。不要依据这份小数据集反复调整 alpha、模型或标注后仍称其为独立测试结果。当前没有实施调参、RRF、RAGAS、LLM 评审或其他 Day 8 功能。
