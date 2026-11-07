# Day 6：检索优化

## 本次文件职责

- `app/query/rewrite.py`：QueryRewriter 接口与 MockQueryRewriter。Mock 只去掉首尾空白，空查询报错；没有真实 LLM 改写。
- `app/query/__init__.py`：导出两个改写类。
- `app/retrieval/hybrid.py`：中文/英文分词、BM25Retriever、HybridRetriever、独立 Min-Max 函数和 HybridSearchResult 调试字段。
- `app/retrieval/reranker.py`：Reranker 接口与 MockReranker。输入为 query 与 list[Document]，输出候选 Document 的重排列表；Mock 原样保序。
- `app/retrieval/text_retriever.py`：保留原向量检索，新增公开 documents 与可选 top_k，便于全语料对齐。
- `app/retrieval/__init__.py`：导出 BM25、Hybrid 和重排接口。
- `app/pipeline.py`：可注入改写器、重排器，普通与流式问答共用新流程；旧的 RAGPipeline(retriever, llm) 仍有效。
- `main.py`：三种检索模式；默认 hybrid、alpha=0.7、candidate-k=10、top-k=3。默认注入两种 Mock。
- `evaluate_retrieval.py`：支持三模式的 Recall@1/3/5，默认仍为 vector，便于与 Day 3 基线比较。
- `requirements.txt`：新增 rank-bm25==0.2.2、jieba==0.42.1。
- `tests/test_query_rewrite.py`、`tests/test_hybrid.py`、`tests/test_reranker.py`：验证接口、词项命中、加权公式、边界和编排顺序。
- `README.md` 与本文件：更新使用方式与设计说明。

## Vector Search 为什么不是万能的

Embedding 将文本压缩成向量，适合寻找含义相近的表达，但可能区分不好罕见产品型号、错误码、数字、专有名词。例如 ERR-42 与 ERR-43 在上下文中很相似，用户通常只想找其中一个。检索效果也依赖模型、切分方式和语料；FAISS 精确计算最近向量，不代表语义模型一定把正确文本放在最近位置。

## BM25 与分词

BM25 根据查询词在文档中的出现频率、稀有程度和文档长度计算相关性。这里使用 rank-bm25 提供的 BM25L 变体，其 IDF 为：

```text
IDF(term) = log(N + 1) - log(df(term) + 0.5)
```

N 是文档数，df 是含该词的文档数。这个 IDF 保持正值，对当前很小的语料也能提供词项区分；它与 BM25Okapi 不是同一公式，比较实验时应记录具体变体。其余 TF 与长度平滑由库实现，没有偷偷改写库内分数。

文档与查询使用同一套 tokenize：中文经 jieba 分词，英文转小写，ERR-42、v1.5 等标识符保持完整。rank-bm25 本身不做这些预处理。中文新词仍可能切分不理想，后续可以用真实查询评估并加入领域词典。

BM25Retriever.search 返回 SearchResult(document, score, rank)，score 为原始 BM25L 分数。空索引、无有效词项或查询无任何匹配时返回空列表；不把 0 分文档当匹配结果返回。top_k 超过数量时返回现有匹配文档，非法 top_k 报 ValueError。

## 为什么需要 Hybrid Search

向量检索擅长语义匹配，BM25 提供词项匹配信号。融合让两类证据共同决定排序，但是否更好需要评估，不能只凭加入一个模块就宣称提升。

cosine 与 BM25 的原始分数尺度不同，直接相加可能让 BM25 数值占主导。因此先分别收集两路实际返回的结果，各自计算 Min-Max，再按 Document 对齐。缺失文档不参与该路 min/max 计算，避免虚构的 0 改变负向量分数的范围：

```text
normalized = (score - min_score) / (max_score - min_score)
final_score = alpha * normalized_vector + (1 - alpha) * normalized_bm25
```

边界策略明确约定如下，它不是零分母公式推导出的唯一答案：

- 一侧无结果：归一化列表为空；合并时该侧原始分数为 None，归一化贡献为 0。
- 一侧只有一个结果，或全部分数相同：已召回结果统一设为 1，保留该侧相同的贡献，避免除零。这不代表置信度为 100%。
- 两侧都无结果：返回 []，不会用未召回文档凑数。
- 只有一侧有结果：保留原来的 alpha 权重，不自动把剩余侧权重改成 1。
- alpha=1/0：只保留对应有效分支的候选；另一侧仍计算分数供调试，其独有候选不参与最终排序。

这里的 Min-Max 与 embedding 的 L2 归一化用途不同：L2 处理向量，使内积等于 cosine；Min-Max 处理检索分数，使两路数值尺度可比较。

手算示例，alpha=0.7，BM25 实际只返回 A、C：

| 文档 | 原 vector | 原 BM25 | vector 归一化 | BM25 归一化 | 融合分 |
|---|---:|---:|---:|---:|---:|
| A | 0.2 | 20 | 0 | 1 | 0.30 |
| B | 0.8 | None | 1 | 0 | 0.70 |
| C | 0.5 | 10 | 0.5 | 0 | 0.35 |

最终 B、C、A。alpha=1 只使用向量信号，alpha=0 只使用 BM25 信号；alpha 必须是 [0,1] 内有限数字。相同分数按原语料顺序稳定排序。归一化后的分数是相对相关性，不是概率，也不能直接用于跨查询置信度比较。

教学版向每一路请求最多 N 个结果，再按实际返回集合归一化、融合和截断。两路语料非空时必须使用同一顺序的同一批 Document 对象；允许某一侧索引为空。按对象身份对应的行号对齐，内容相同但 chunk_id 不同的块不会被合并。建库后不应修改语料内容。

## 完整分数数据流与调试字段

```text
query
  ├→ VectorRetriever.search → 原始向量结果 → 独立 Min-Max
  └→ BM25Retriever.search   → 原始 BM25 结果 → 独立 Min-Max
      → 按 Document 对齐有效候选
      → 缺失侧贡献设为 0
      → 0.7 × normalized_vector_score + 0.3 × normalized_bm25_score
      → final_score 降序排序，同分保留语料顺序
      → Top-K → 更新 rank → HybridSearchResult
```

每个 HybridSearchResult 包含 document、vector_score、bm25_score、normalized_vector_score、normalized_bm25_score、final_score 和 rank。score 是 final_score 的只读兼容属性；向量/BM25 原有 SearchResult 和 search 接口不变。pipeline 通过 dataclasses.replace 更新 rank 时会保留所有调试字段，Context Builder 的对象和字典两种输入都兼容。

```python
from dataclasses import asdict

for result in hybrid.search("FAISS 是什么？", top_k=3):
    print(result.document.metadata, result.vector_score, result.bm25_score)
    print(result.normalized_vector_score, result.normalized_bm25_score, result.final_score)
    payload = asdict(result)  # document 也会递归转成含 text/metadata 的字典
```

## Min-Max 的局限

1. 极端最大值会压缩其他结果，例如 [1, 2, 100] 变成约 [0, 0.01, 1]。
2. 很小的原始差异也可能被拉成 0 和 1，因此近似等分时会放大噪声。
3. min/max 依赖当前结果集合；候选数量或文档变化可能改变同一文档的融合分。
4. 缩放统一数值范围，不会校准两路分数的概率含义或分布，也不保证 alpha 对应真实可靠性。
5. 单结果/等分时需要额外策略；本项目设为 1，只表示等贡献，不能据此判断答案存在或正确。

后续应使用标注集选择 alpha，并比较融合方式；跨查询的拒答阈值不能只凭当前 Min-Max 分数决定。

## Retriever 和 Reranker 的区别

Retriever 从知识库找候选；Reranker 对已有候选做更细的相关性排序。重排无法找回没有召回的文档，所以 candidate-k 通常大于最终 top-k。本次默认 10 个候选、最后 3 个。

MockReranker 仅返回候选列表副本，不改变顺序和 Document，不是真实相关性模型。未来 Cross Encoder 可以联合编码 query 和 document.text，再按相关性分数排序返回同一批原始 Document 对象。

pipeline 验证重排结果只能是候选对象的无重复子集，避免外来文本混入。使用 dataclasses.replace 更新 rank，保留原检索 score 与 metadata，不修改原始结果。当前 context 的 score 仍是检索/融合分，不是 Cross Encoder 分数。

## Query Rewrite 解决什么问题

查询表达可能口语化、过短、缩写过多，或与知识库措辞差异很大。改写器可以将意图转为更适合检索的表达，但必须防止引入新事实或改变原意。当前 Mock 只有空白清理，不做同义扩展、指代消解或 LLM 调用；若要消解多轮指代，还需要显式提供对话上下文。

改写结果仅用于检索。Reranker 与最终回答都使用用户原问题，保留真实意图。真实 LLM 改写器只需实现 rewrite(query)，真实 Cross Encoder 重排器只需实现 rerank(query, documents)，pipeline 不需要知道模型细节。

## 当前完整数据流

```text
启动：文档 → Loader → Splitter → 相同的 Document chunks
                                  ├→ Embedding → FAISS
                                  └→ 分词 → BM25L

POST /chat 或 /chat/stream
  → 用户原问题
  → QueryRewriter（当前 Mock）
  → 改写后的检索 query
  → Vector + BM25
  → 分数归一化 → alpha 加权 → candidate-k 候选
  → Reranker(原问题, 候选 Documents)（当前 Mock）
  → 最终 top-k → 更新 rank，保留检索 score
  → Context Builder
  → Prompt Builder(原问题, context)
  → LLM.generate / stream_generate
  → JSON answer / SSE 文本增量
```

选 vector 或 bm25 模式时替换中间的检索分支，其余编排相同。BM25 模式无需加载 embedding 模型。

## 复杂度与验证

设 N 是 chunk 数，d 是向量维度，t 是查询词项数，M 是重排候选数。

- 本版 BM25 库逐词扫描各文档词频，查询约 O(tN)，完整排序 O(N log N)。
- Hybrid 对全语料做向量检索 O(Nd)，再对全分数融合 O(N)，完整排序存在 O(N log N) 开销。它是正确性清楚的教学实现，规模变大后应改为有限候选集合并，并重新评估融合方式。
- Mock 改写为字符串清理，约 O(查询长度)；Mock 重排复制列表 O(M)。真实 Cross Encoder 成本还取决于模型和文本长度，不能用 Mock 耗时推断。

```powershell
.\.venv\Scripts\python.exe main.py --mock --retrieval hybrid --alpha 0.7 --candidate-k 10 --top-k 3
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval vector
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval bm25
.\.venv\Scripts\python.exe evaluate_retrieval.py --retrieval hybrid --alpha 0.7
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

原 8 文档、6 查询的微型样本只用于验证流程。应添加错误码、专有名词、同义表达等更有区分度的人工标注问题，再调整 alpha、candidate-k 和分词参数。两个 Mock 本身不提供效果提升。

依赖依据：[rank-bm25 使用方式](https://github.com/dorianbrown/rank_bm25)、[BM25L 源码](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)、[jieba 分词](https://github.com/fxsjy/jieba)。
