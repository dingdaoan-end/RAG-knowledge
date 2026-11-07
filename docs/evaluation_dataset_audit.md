# 评测集独立审核报告：RAG 求职冲刺 Day 2 · 第一阶段

日期：2026-10-26（Asia/Shanghai）。报告版本：audit-v1。执行者：Codex（代码与正文预核查）。人工审核人：尚未指定。

> 本报告不是已人工签核的新 Ground Truth。19 道题的审核状态全部为 pending；没有因 Codex 检查而自动标为 verified。参考答案、标签、数据版本和历史实验均保持原样。

## 1. 实际检查文件、结构与版本

本轮阅读全部 18 条正文，并逐题对照完整语料，而不是只看问题名或 ID。没有调用真实 LLM，没有读取密钥文件，没有运行默认评估命令，也没有用历史实验分数决定标签。

实际阅读的文件：

- [data/evaluation/retrieval_eval.json](D:/RAGrepository/data/evaluation/retrieval_eval.json)：全部语料、问题、等级和元数据。
- [data/evaluation/qa_eval.json](D:/RAGrepository/data/evaluation/qa_eval.json)：全部问题、答案、事实、引用、answerable。
- [docs/day7_evaluation.md](D:/RAGrepository/docs/day7_evaluation.md)：标注规则、指标定义、候选预算及限制。
- [docs/day8_qa_evaluation.md](D:/RAGrepository/docs/day8_qa_evaluation.md)：QA 数据约束、人工评审与证据不足定义。
- [app/evaluation/dataset.py](D:/RAGrepository/app/evaluation/dataset.py)：稳定 ID、正文、标签、重复字段及指纹校验。
- [app/evaluation/qa_dataset.py](D:/RAGrepository/app/evaluation/qa_dataset.py)：必填字段、事实清单、文档映射、SHA 绑定。
- [app/evaluation/retrieval_metrics.py](D:/RAGrepository/app/evaluation/retrieval_metrics.py)：Recall、RR/MRR、分级 DCG/nDCG 及边界规则。
- [app/evaluation/evaluator.py](D:/RAGrepository/app/evaluation/evaluator.py)：同规则四方案、结果映射、候选约束和报告防覆盖。
- [app/ingestion/document.py](D:/RAGrepository/app/ingestion/document.py)：Document 的 text/metadata，不重构模型。
- [app/embedding/embedder.py](D:/RAGrepository/app/embedding/embedder.py)：只静态核对 q03 默认模型及归一化，未加载模型。
- [scripts/evaluate_retrieval.py](D:/RAGrepository/scripts/evaluate_retrieval.py)：确认默认数据路径，未执行评估。
- [tests/test_retrieval_metrics.py](D:/RAGrepository/tests/test_retrieval_metrics.py)：阅读并运行指标和边界测试。
- [tests/test_retrieval_evaluator.py](D:/RAGrepository/tests/test_retrieval_evaluator.py)：检查断言并运行映射、候选、报告保护测试。
- [tests/test_qa_evaluation.py](D:/RAGrepository/tests/test_qa_evaluation.py)：检查断言并运行 QA、负例、审核及冻结保护测试。

只对 data/evaluation 与 results 下既有 13 个文件计算 SHA256 以作前后保护检查；历史回答评分没有被沿用为本次数据集的人工状态。

| 数据集 | 实际版本 | 结构/数量 | 原始审核声明 |
|---|---|---|---|
| rag_knowledge_teaching_v1 | metadata.version=1 | metadata/documents/queries；18 Chunk、14 Query；26 个显式标签（22 正相关、4 个 0） | teaching_labels_pending_human_review |
| rag_qa_teaching_v1 | version=1.0 | version/corpus_sha256/metadata/samples；5 QA、15 项 required_facts、7 次文档引用；3 true、2 false | teaching_references_pending_human_review |

文件路径、字段和题目数量与任务描述一致。另有 data/retrieval_eval.json 是旧脚本数据；本轮正式数据不引用它，不能混为同一 Ground Truth。

检索数据 SHA256：085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71

QA 数据 SHA256：9e33ac59dd0ba406a047249442672a5a01e22ef73f8f4d3c4eb054133d89c1eb

QA corpus_sha256 与检索数据实际字节指纹一致，不能只看版本号。

## 2. 原始规则与证据映射

### 2.1 相关性等级（原样保留）

| 等级 | 原始定义 |
|---|---|
| 0 | 不相关或不能回答问题 |
| 1 | 提供部分背景或限制，有一定帮助 |
| 2 | 相关，提供重要但不完整的回答 |
| 3 | 高度相关，直接回答核心问题 |

- 等级 1/2/3 都算二元正相关；未列出文档按 0。弱相关边界和漏标直接影响 Recall 分母、首个相关名次与理想排序。
- 排名从 1 开始；跨 Query 宏平均。Recall@K=命中正例数/全部正例数；RR=首个正例排名倒数，无命中为 0。
- 既有统一候选预算为 10，实际报告 MRR@10，不能称为不截断全库 MRR。
- DCG 使用 (2^rel-1)/log2(rank+1)，nDCG 除以全部标注的理想 DCG；等级 3/2/1 的增益分别为 7/3/1，不是二元 Recall。
- 14 个检索题都有正相关标签，格式支持这些指标；格式可计算不等于标签已经人工可信。无正例会报错，不能把 QA 负例直接放进当前检索指标后静默记 0。
- QA answerable 原始定义：true 表示已有资料能回答所请求的事实；false 表示所请求的事实或确定结论无法由现有资料支持，期望说明证据限制，而不是猜测。

### 2.2 映射实际结果

18 个稳定 ID 均能映射到 documents[].text，且正文非空。Loader 将 id 放入 Document.metadata 的 eval_id，并保留 source/chunk_id；未使用 Python 对象内存地址作为持久标识。26 个检索标签引用、7 个 QA 引用全部存在，没有重复问题 ID、重复文档 ID 或重复 source/chunk_id。

source 的 rag.md、faiss.md 等 10 个名称是合成语料的逻辑来源。在项目根目录、data、data/evaluation 的对应位置未发现同名独立文件。此事实不构成正文映射失败：原始评估正文内嵌在冻结 JSON 中，以下逐题提供绝对路径行号，附录列出全部原文。但它也不能当作官方出处、原文件来源链或真实用户日志已经核验。

### 2.3 审核状态与计数

状态约定：pending=待人工核验；verified=人工核验通过；needs_revision=确认需修订；insufficient_evidence=材料不足以判断。没有新的人工签核，19 条记录全部 pending。建议结论单独记录，不冒充人工状态。

| 口径 | 数量 | 解释 |
|---|---:|---|
| 正文与映射预核查完成 | 19/19 | 14 检索 + 5 QA |
| 核心解答或负例限制有充分语料依据 | 19/19 | 仅为冻结语料内部支持，不表示所有标签已通过人工或外部事实核验 |
| 未发现重点标签争议的预核查题 | 16 | 11 检索 + 5 QA，仍待人工签核 |
| 标签边界/漏标重点候选 | 3 | q02、q04、q13，不是已确认错误 |
| 已确认必须修订 | 0 | 没有替用户裁决争议或改标 |
| 因正文缺失/映射失败无法判断 | 0 | 两个有效 QA 负例不属于审核缺证 |
| pending / verified / needs_revision / insufficient_evidence | 19 / 0 / 0 / 0 | 人工完成数为 0 |

## 3. 检索集逐题审核记录

检索集没有参考答案字段；只列出现有分级标注，不补造标准答案。每题已与全部 18 条正文作对照；未发现明显漏标不等于多人一致标注保证。

### q01

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q01](D:/RAGrepository/data/evaluation/retrieval_eval.json:34)。
- 原始问题：RAG 为什么可以降低大模型幻觉？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| rag_evidence | 3 | [rag_evidence](D:/RAGrepository/data/evaluation/retrieval_eval.json:15)；rag.md / 1；完整正文见附录 |
| rag_overview | 2 | [rag_overview](D:/RAGrepository/data/evaluation/retrieval_eval.json:14)；rag.md / 0；完整正文见附录 |
| rag_limits | 1 | [rag_limits](D:/RAGrepository/data/evaluation/retrieval_eval.json:16)；rag.md / 2；完整正文见附录 |
| rag_terms | 0 | [rag_terms](D:/RAGrepository/data/evaluation/retrieval_eval.json:17)；rag.md / 3；完整正文见附录 |

- 核心证据充分性：问题清晰，直接询问降低幻觉的机制。rag_evidence 说明外部事实/参数记忆机制；rag_overview 说明检索结果进入提示词；rag_limits 给出限制，三者共同支持核心回答。
- 相关性等级依据：3/2/1 分别对应直接解释、流程补充、适用限制。rag_terms 虽有相同关键词，但明确不解释机制，0 有正文依据。
- 全库漏标/关键词误标检查：全库对照未发现另一个解释幻觉降低机制的未标注 Chunk；不因其他文档提到 RAG 就算相关。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：确认限制性背景 rag_limits 算等级 1，而非要求每个正例都直接回答为什么。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q02

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q02](D:/RAGrepository/data/evaluation/retrieval_eval.json:35)。
- 原始问题：FAISS 的内积搜索在什么条件下等价于余弦相似度？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| faiss_ip | 3 | [faiss_ip](D:/RAGrepository/data/evaluation/retrieval_eval.json:18)；faiss.md / 0；完整正文见附录 |
| cosine_unit | 3 | [cosine_unit](D:/RAGrepository/data/evaluation/retrieval_eval.json:19)；faiss.md / 1；完整正文见附录 |
| embedding_bge | 1 | [embedding_bge](D:/RAGrepository/data/evaluation/retrieval_eval.json:20)；embedding.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰；faiss_ip 和 cosine_unit 都直接写出查询/文档向量单位归一化及内积等于余弦，核心条件有充分证据。
- 相关性等级依据：两个 3 有直接依据。embedding_bge 只说语义向量和预计算，没有 L2、单位长度或余弦条件；一般背景是否算 1 存在边界争议，不能断定当前标签已错。
- 全库漏标/关键词误标检查：其余正文没有另一个明确说明该条件的 Chunk。争议是现有 embedding_bge:1，不是按它是否被检索命中改标。
- 建议结论：重点人工裁决；暂不修订。
- 需要人工确认：请决定一般 Embedding 背景是否达到等级 1。归一化是项目采用的标准充分条件，不将其扩张为所有可能排序等价情况的唯一数学必要条件。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q03

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q03](D:/RAGrepository/data/evaluation/retrieval_eval.json:36)。
- 原始问题：本项目用什么模型把中文文本编码成语义向量？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| embedding_bge | 3 | [embedding_bge](D:/RAGrepository/data/evaluation/retrieval_eval.json:20)；embedding.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰但依赖项目版本。embedding_bge 明确给出 sentence-transformers 和 BAAI/bge-small-zh-v1.5，直接支持答案。
- 相关性等级依据：embedding_bge:3 直接回答模型名称；app/embedding/embedder.py:10–16 的默认实现与正文一致，仅静态核对，没有加载模型。
- 全库漏标/关键词误标检查：其余 17 条正文没有另一个本项目中文向量模型名称，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：确认本项目指冻结语料/当前代码默认配置；不能保证所有自定义启动参数都使用此模型。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q04

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q04](D:/RAGrepository/data/evaluation/retrieval_eval.json:37)。
- 原始问题：示例日志里 CUDA error 1175 应该根据什么线索排查？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| cuda_1175 | 3 | [cuda_1175](D:/RAGrepository/data/evaluation/retrieval_eval.json:25)；cuda_examples.md / 0；完整正文见附录 |
| cuda_1176 | 0 | [cuda_1176](D:/RAGrepository/data/evaluation/retrieval_eval.json:26)；cuda_examples.md / 1；完整正文见附录 |
| cuda_oom | 0 | [cuda_oom](D:/RAGrepository/data/evaluation/retrieval_eval.json:27)；cuda_examples.md / 2；完整正文见附录 |

- 核心证据充分性：问题明确限定示例日志。cuda_1175 给出 driver initialization failed、完整日志及驱动/运行时/框架版本，直接支持排查线索。
- 相关性等级依据：cuda_1175:3 有据；cuda_1176 的设备序号建议不能替代 1175。cuda_oom:0 有争议，其正文末尾实际包含“1175 的驱动初始化线索”，可能提供等级 1 的区别信息。
- 全库漏标/关键词误标检查：没有发现另一条未标注的直接解答。争议来自已经显式标 0 的 cuda_oom，不能忽略正文中的正确线索，也不能因编号重合就标高相关。
- 建议结论：重点人工裁决 cuda_oom:0；可能需修订，尚未确认。
- 需要人工确认：请判断给出正确区别并提及驱动初始化是否算 1；确认 1175/1176 不被当作官方错误码定义。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q05

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q05](D:/RAGrepository/data/evaluation/retrieval_eval.json:38)。
- 原始问题：示例 CUDA error 1176 对应的日志线索和排查对象是什么？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| cuda_1176 | 3 | [cuda_1176](D:/RAGrepository/data/evaluation/retrieval_eval.json:26)；cuda_examples.md / 1；完整正文见附录 |
| cuda_1175 | 0 | [cuda_1175](D:/RAGrepository/data/evaluation/retrieval_eval.json:25)；cuda_examples.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰，要求日志线索与排查对象。cuda_1176 同时给出 device index out of range、设备序号及可见 GPU 列表。
- 相关性等级依据：cuda_1176:3 直接回答；cuda_1175:0 没有解释 1176 的设备范围问题，不能因 CUDA/error 词项相同算相关。
- 全库漏标/关键词误标检查：全库没有另一个提供 1176 具体排查对象的 Chunk；cuda_oom 的显存建议不回答设备索引问题。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：只适用于该示例日志，不作为真实错误编号的通用解释。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q06

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q06](D:/RAGrepository/data/evaluation/retrieval_eval.json:39)。
- 原始问题：CUDA out of memory 时如何减少 GPU 显存占用？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| cuda_oom | 3 | [cuda_oom](D:/RAGrepository/data/evaluation/retrieval_eval.json:27)；cuda_examples.md / 2；完整正文见附录 |

- 核心证据充分性：问题清晰。cuda_oom 提供减少 batch size、缩短输入、释放张量及检查其他进程显存占用的措施。
- 相关性等级依据：cuda_oom:3 直接回答减少显存占用；其他 CUDA 初始化和设备索引材料不能替代。
- 全库漏标/关键词误标检查：全库没有另一条补充 OOM 减占用方法的正文，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：这些是初步措施，不是所有真实 OOM 的保证修复命令。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q07

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q07](D:/RAGrepository/data/evaluation/retrieval_eval.json:40)。
- 原始问题：切块时为什么保留 chunk_overlap，它与 chunk_size 有什么约束？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| chunk_overlap | 3 | [chunk_overlap](D:/RAGrepository/data/evaluation/retrieval_eval.json:21)；chunking.md / 0；完整正文见附录 |

- 核心证据充分性：问题同时询问原因和约束，chunk_overlap 同时解释减少边界截断与 overlap 小于 chunk_size，覆盖完整。
- 相关性等级依据：chunk_overlap:3 直接回答两个子问题。
- 全库漏标/关键词误标检查：其他正文偶尔提到切块，但不说明 overlap 动机或约束，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：本题不要求完整参数校验清单；不要临时增加参考正文未包含的必答细节。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q08

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q08](D:/RAGrepository/data/evaluation/retrieval_eval.json:41)。
- 原始问题：错误编号或产品型号必须精确匹配时，BM25 为什么有帮助？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| bm25_terms | 3 | [bm25_terms](D:/RAGrepository/data/evaluation/retrieval_eval.json:22)；retrieval.md / 0；完整正文见附录 |
| hybrid_fusion | 1 | [hybrid_fusion](D:/RAGrepository/data/evaluation/retrieval_eval.json:23)；retrieval.md / 1；完整正文见附录 |

- 核心证据充分性：问题具有精确词项检索意义。bm25_terms 说明词项匹配、错误编号/型号和向量混淆；hybrid_fusion 说明两路互补。
- 相关性等级依据：bm25_terms:3 直接说明帮助，hybrid_fusion:1 为方法背景。日志中出现编号不等于解释 BM25，因此不直接把 CUDA 文档算正例。
- 全库漏标/关键词误标检查：其余正文没有另一条说明 BM25 精确词项优势的解答，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：有帮助不等于保证首名命中或严格字符串匹配；分词和语料仍影响结果。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q09

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q09](D:/RAGrepository/data/evaluation/retrieval_eval.json:42)。
- 原始问题：Hybrid 为什么不能直接相加向量分数和 BM25 分数？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| hybrid_fusion | 3 | [hybrid_fusion](D:/RAGrepository/data/evaluation/retrieval_eval.json:23)；retrieval.md / 1；完整正文见附录 |
| bm25_terms | 1 | [bm25_terms](D:/RAGrepository/data/evaluation/retrieval_eval.json:22)；retrieval.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰。hybrid_fusion 明确原始分数范围不同、分别 Min-Max 再融合；bm25_terms 说明词项评分背景。
- 相关性等级依据：hybrid_fusion:3 是直接理由，bm25_terms:1 只提供一侧评分背景，未被错误标作完整解答。
- 全库漏标/关键词误标检查：其他正文没有补充两路分数尺度/融合规则，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：本题核对当前方案的分数可比性，不要求证明 Min-Max 永远最优或引入新算法。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q10

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q10](D:/RAGrepository/data/evaluation/retrieval_eval.json:43)。
- 原始问题：为什么不能提前为文档计算一个固定的 Cross-Encoder 重排分数？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| reranker_pairs | 3 | [reranker_pairs](D:/RAGrepository/data/evaluation/retrieval_eval.json:24)；retrieval.md / 2；完整正文见附录 |
| embedding_bge | 1 | [embedding_bge](D:/RAGrepository/data/evaluation/retrieval_eval.json:20)；embedding.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰。reranker_pairs 说明联合编码、分数依赖具体问题；embedding_bge 的文档向量可提前算构成有用对照。
- 相关性等级依据：reranker_pairs:3 直接回答；embedding_bge:1 对比可预计算向量与问题相关重排分数，有实质背景作用。
- 全库漏标/关键词误标检查：其他正文没有 Cross-Encoder 固定分数的补充机制，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：不可预计算统一分数不等于模型权重、文档文本或向量不能缓存。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q11

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q11](D:/RAGrepository/data/evaluation/retrieval_eval.json:44)。
- 原始问题：Transformer 的 Self-Attention 如何建模 Token 之间的关系？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| transformer_attention | 3 | [transformer_attention](D:/RAGrepository/data/evaluation/retrieval_eval.json:28)；transformer.md / 0；完整正文见附录 |

- 核心证据充分性：问题自然但较概括。transformer_attention 给出 Query/Key/Value 交互、缩放点积及 softmax，足以作简化机制解释。
- 相关性等级依据：transformer_attention:3 直接解释 Token 关系；没有把词频匹配背景当作注意力机制答案。
- 全库漏标/关键词误标检查：其他正文没有 Self-Attention 解答，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：确认只要求概念说明；现有正文不足以要求矩阵维度、掩码或多头推导等扩展细节。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q12

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q12](D:/RAGrepository/data/evaluation/retrieval_eval.json:45)。
- 原始问题：扫描版 PDF 提取不到文字时，入库之前需要做什么？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| pdf_ocr | 3 | [pdf_ocr](D:/RAGrepository/data/evaluation/retrieval_eval.json:30)；ingestion_notes.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰。pdf_ocr 说明扫描页可能无文本层，需先 OCR 再清洗、切块和 Embedding。
- 相关性等级依据：pdf_ocr:3 直接回答入库前处理。
- 全库漏标/关键词误标检查：其他正文没有扫描 PDF 的补充方案，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：知识性问答不表示当前 Loader 已实现 PDF/OCR 支持，不能据此作功能承诺。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q13

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q13](D:/RAGrepository/data/evaluation/retrieval_eval.json:46)。
- 原始问题：使用 RAG 是否就能保证完全没有幻觉？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| rag_limits | 3 | [rag_limits](D:/RAGrepository/data/evaluation/retrieval_eval.json:16)；rag.md / 2；完整正文见附录 |
| rag_evidence | 2 | [rag_evidence](D:/RAGrepository/data/evaluation/retrieval_eval.json:15)；rag.md / 1；完整正文见附录 |

- 核心证据充分性：问题明确是否保证没有幻觉。rag_limits 直接否定并列出失败原因；rag_evidence 说明降低幻觉的机制。
- 相关性等级依据：rag_limits:3 直接回答；rag_evidence:2 提供重要机制背景但不完整解释全部限制。
- 全库漏标/关键词误标检查：rag_overview 未列出、按 0，正文有 RAG 流程背景但没说不能保证。结合 q01 的背景/限制也算正例及等级 1 的宽定义，存在弱相关漏标候选；不能直接断言应补正例。
- 建议结论：重点人工裁决 rag_overview 是否漏标；不自动补标签。
- 需要人工确认：请统一只解释流程的背景是否达到等级 1，不按哪种检索方案受益决定。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

### q14

- 数据集：rag_knowledge_teaching_v1；原始问题位置：[q14](D:/RAGrepository/data/evaluation/retrieval_eval.json:47)。
- 原始问题：在 PowerShell 中怎样使用项目 .venv 的 Python 启动 main.py？
- 审核状态：pending；人工审核人/日期：未填写。

| 当前相关文档 ID | 当前 relevance | 原始证据位置 / source / chunk_id |
|---|---:|---|
| python_venv | 3 | [python_venv](D:/RAGrepository/data/evaluation/retrieval_eval.json:29)；setup.md / 0；完整正文见附录 |

- 核心证据充分性：问题清晰，python_venv 给出 PowerShell 命令 .\.venv\Scripts\python.exe main.py，并说明当前目录及同一依赖环境。
- 相关性等级依据：python_venv:3 直接回答启动方式。
- 全库漏标/关键词误标检查：其他正文没有另一条虚拟环境启动命令，无明显漏标。
- 建议结论：建议保留，待人工签核。
- 需要人工确认：命令前提是项目根目录且 .venv 存在；不保证 API 配置和服务运行都成功。
- 指标适用性：有正相关 Ground Truth，可按相同标签计算 Recall@K、MRR@候选预算和 nDCG@K；仍受上述标签质量限制。

## 4. QA 集逐题审核记录

### qa_001

- 数据集：rag_qa_teaching_v1；原始问题位置：[qa_001](D:/RAGrepository/data/evaluation/qa_eval.json:14)。
- 原始问题：RAG 为什么可以降低大模型幻觉？它能保证完全消除幻觉吗？
- 审核状态：pending；人工审核人/日期：未填写。
- 当前 answerable：true。
- 当前 relevant_document_ids：["rag_evidence", "rag_limits", "rag_overview"]。
- 当前参考答案（原样）：

> RAG 检索外部知识，为生成模型提供相关事实和证据，减少仅凭参数记忆编造答案的情况。但检索错误、资料过时或模型误读仍会导致错误，因此不能保证完全消除幻觉。

| 当前 required_fact（原样） | 实际支持位置 / 说明 |
|---|---|
| 检索外部知识 | rag_overview + rag_evidence |
| 向生成模型提供相关事实和证据 | rag_evidence + rag_overview |
| 不能保证完全消除幻觉 | rag_limits |

- 对应原始证据：[rag_evidence](D:/RAGrepository/data/evaluation/retrieval_eval.json:15)、[rag_limits](D:/RAGrepository/data/evaluation/retrieval_eval.json:16)、[rag_overview](D:/RAGrepository/data/evaluation/retrieval_eval.json:14)
- 参考答案证据充分性：三篇正文共同支持参考答案。rag_overview 支持检索外部知识和提示词流程，rag_evidence 支持外部事实/参数记忆机制，rag_limits 支持不能保证及检索错误、过时、误读等限制。未发现参考答案超出这三篇的技术断言。
- answerable、必答事实与歧义：answerable=true 与证据一致。三个 required_facts 覆盖核心问题。参考答案中的具体失败原因目前不是独立必答项，这是评分严格度选择，不是已确认漏标。
- 相关性等级：QA 无 0–3 relevance 字段，本轮只审核文档引用，不补造分数。
- 建议结论：参考答案和正/负例标记在冻结语料内有据，建议保留并等待人工签核；未修改。
- 需要人工确认：请确认未来是否额外要求至少一个失败原因；本轮不增加。历史模型回答的人工评分不代替本轮标签签核。

### qa_002

- 数据集：rag_qa_teaching_v1；原始问题位置：[qa_002](D:/RAGrepository/data/evaluation/qa_eval.json:22)。
- 原始问题：FAISS IndexFlatIP 在什么条件下等价于余弦相似度检索？
- 审核状态：pending；人工审核人/日期：未填写。
- 当前 answerable：true。
- 当前 relevant_document_ids：["faiss_ip", "cosine_unit"]。
- 当前参考答案（原样）：

> 查询向量和文档向量都需要做 L2 归一化，使范数为 1；此时内积等于余弦相似度，IndexFlatIP 才能按余弦相似度进行检索。

| 当前 required_fact（原样） | 实际支持位置 / 说明 |
|---|---|
| 查询向量和文档向量都需要 L2 归一化 | faiss_ip + cosine_unit |
| 归一化后的范数为 1 | cosine_unit |
| 单位向量的内积等于余弦相似度 | faiss_ip + cosine_unit |

- 对应原始证据：[faiss_ip](D:/RAGrepository/data/evaluation/retrieval_eval.json:18)、[cosine_unit](D:/RAGrepository/data/evaluation/retrieval_eval.json:19)
- 参考答案证据充分性：faiss_ip 与 cosine_unit 逐项支持双侧 L2 归一化、范数为 1 和内积等于余弦，参考答案与正文一致。
- answerable、必答事实与歧义：answerable=true、两条引用及三项事实有据。都需要/才应理解为本项目的标准充分条件，而非所有排序恰好等价情形的必要条件分类。
- 相关性等级：QA 无 0–3 relevance 字段，本轮只审核文档引用，不补造分数。
- 建议结论：参考答案和正/负例标记在冻结语料内有据，建议保留并等待人工签核；未修改。
- 需要人工确认：确认项目实现语境是否明确；若未来要考数学必要条件或零向量，应有独立证据和新版本，不擅改参考答案。

### qa_003

- 数据集：rag_qa_teaching_v1；原始问题位置：[qa_003](D:/RAGrepository/data/evaluation/qa_eval.json:30)。
- 原始问题：Cross-Encoder 为什么不能提前为每篇文档计算一个适用于所有问题的固定重排分数？
- 审核状态：pending；人工审核人/日期：未填写。
- 当前 answerable：true。
- 当前 relevant_document_ids：["reranker_pairs"]。
- 当前参考答案（原样）：

> Cross-Encoder 将问题和文档组成文本对联合编码，相关性分数同时依赖问题与文档。同一篇文档面对不同问题会得到不同分数，因此不能预计算适用于所有问题的固定分数。它只重排召回候选。

| 当前 required_fact（原样） | 实际支持位置 / 说明 |
|---|---|
| 联合编码 query 与 document 文本对 | reranker_pairs |
| 相关性分数依赖具体问题 | reranker_pairs |
| 不能预计算适用于所有问题的固定重排分数 | reranker_pairs |

- 对应原始证据：[reranker_pairs](D:/RAGrepository/data/evaluation/retrieval_eval.json:24)
- 参考答案证据充分性：reranker_pairs 逐句支持联合编码、分数随问题变化、不能统一预计算，以及只重排候选。没有根据名称猜测正文。
- answerable、必答事实与歧义：answerable=true 与证据一致。三项事实回答为何不能固定预计算；参考答案的只重排候选有据，但不作为独立必答项。
- 相关性等级：QA 无 0–3 relevance 字段，本轮只审核文档引用，不补造分数。
- 建议结论：参考答案和正/负例标记在冻结语料内有据，建议保留并等待人工签核；未修改。
- 需要人工确认：确认本题必答项围绕预计算问题，不额外要求完整两阶段检索架构。

### qa_004

- 数据集：rag_qa_teaching_v1；原始问题位置：[qa_004](D:/RAGrepository/data/evaluation/qa_eval.json:38)。
- 原始问题：只看到 CUDA error 1175，能否确定我的显卡已经损坏，并给出唯一的修复命令？
- 审核状态：pending；人工审核人/日期：未填写。
- 当前 answerable：false。
- 当前 relevant_document_ids：["cuda_1175"]。
- 当前参考答案（原样）：

> 不能仅凭这一编号确定显卡损坏，也不能给出唯一修复命令。知识库只有示例日志线索，缺少用户的完整日志和环境信息。应明确说明证据不足，并请求完整错误日志及驱动、运行时和框架版本。

| 当前 required_fact（原样） | 实际支持位置 / 说明 |
|---|---|
| 不能仅凭错误编号认定显卡损坏 | cuda_1175；全库缺少硬件损坏证据 |
| 证据不足，不能给出唯一修复命令 | cuda_1175；全库没有唯一修复命令 |
| 需要完整日志和环境版本信息 | cuda_1175 |

- 对应原始证据：[cuda_1175](D:/RAGrepository/data/evaluation/retrieval_eval.json:25)
- 参考答案证据充分性：cuda_1175 明确是示例，编号不足以确定修复方法，需完整日志和版本。全库 18 条正文没有该用户的硬件检测报告、显卡损坏证据或唯一修复命令；拒绝确定诊断与请求补证有依据。
- answerable、必答事实与歧义：answerable=false 符合定义：无法支持用户索取的具体损坏事实和唯一命令；不是连限制说明都不能回答。三项事实覆盖不判损坏、不造命令和补日志版本。
- 相关性等级：QA 无 0–3 relevance 字段，本轮只审核文档引用，不补造分数。
- 建议结论：参考答案和正/负例标记在冻结语料内有据，建议保留并等待人工签核；未修改。
- 需要人工确认：确认 false 按请求的诊断事实不可确定来解释，而非按是否能回答一个否定句；不将示例编号当官方定义。

### qa_005

- 数据集：rag_qa_teaching_v1；原始问题位置：[qa_005](D:/RAGrepository/data/evaluation/qa_eval.json:46)。
- 原始问题：这台机器当前安装的 GPU 具体型号、驱动版本和 CUDA 版本分别是什么？
- 审核状态：pending；人工审核人/日期：未填写。
- 当前 answerable：false。
- 当前 relevant_document_ids：[]。
- 当前参考答案（原样）：

> 知识库没有当前机器的实际硬件和软件版本信息，无法确定这些值。应说明资料不足，需要用户提供环境检测结果，不能把技术文档或示例日志当作当前机器的事实。

| 当前 required_fact（原样） | 实际支持位置 / 说明 |
|---|---|
| 现有资料不能确定当前机器的型号和版本 | 全库 18 条正文：无当前环境清单 |
| 不编造具体硬件型号或版本号 | 拒绝编造的评估要求；没有正向硬件事实证据 |
| 需要当前环境的实际检测信息 | 当前环境事实缺失，需要用户提供实际检测信息 |

- 对应原始证据：没有正向相关 Chunk；见附录全部 18 条正文的闭集排查，不虚构来源 ID。
- 参考答案证据充分性：逐条读过全部 18 个 Chunk：正文均为通用技术或示例日志，没有当前机器 GPU 型号、驱动号或 CUDA 实测版本。参考答案的资料不足结论在该冻结语料闭集内成立，未对本机作探测。
- answerable、必答事实与歧义：answerable=false 和空相关列表符合现有规则。required_facts 是证据限制与处理要求，不是虚构硬件事实。有效负例不等于审核缺证或映射失败。
- 相关性等级：QA 无 0–3 relevance 字段，本轮只审核文档引用，不补造分数。
- 建议结论：参考答案和正/负例标记在冻结语料内有据，建议保留并等待人工签核；未修改。
- 需要人工确认：确认这台机器指提问者当前环境且任务限定静态知识库，不含系统探测工具；未来加入实际环境清单时需重新审核。

## 5. 需要人工复核的优先清单

所有 19 题都需要你最终签核。优先裁决以下 3 项影响 Ground Truth 一致性的事项，先统一标准，再决定是否另建修订版本。

| 问题 | 实际证据与待裁决事项 |
|---|---|
| q02 / embedding_bge:1 | 正文只说语义向量与预计算，没有单位归一化/余弦条件。一般背景是否达到等级 1？接受与否都需一致解释，不能看召回分数决定。 |
| q04 / cuda_oom:0 | 显式 0 的正文包含“1175 的驱动初始化线索”，是否算有一定帮助？cuda_1176 仅有相似编号并不是改标理由。 |
| q13 / rag_overview 未列出 | 当前按 0；正文提供 RAG 流程背景但没直接否定保证。请判断是否漏掉弱相关背景，并统一与其他题的等级标准。 |

附加事项（尚无确定错标证据）：q03 的版本；q04/q05/qa_004 的合成日志范围；q08 的有帮助而非保证匹配；q11 概念层次；q12 知识内容而非已有功能；qa_001/qa_003 的非必答补充事实；q02/qa_002 的充分条件语境；qa_004 的 false 语义；qa_005 的静态知识库范围。详见每题记录。

本轮是数据集标签审核，Day 8 是模型实际回答的人工评分；历史回答得分不能直接认证整份参考答案、分级标签或全语料。

## 6. 映射失败、证据缺失与主要风险

### 6.1 清单

- 文档 ID 映射失败：0；26 个检索标签引用及 7 个 QA 引用全部可定位。
- 正文缺失：0；18 条正文均已实际阅读。
- 语料绑定失败：0；QA corpus_sha256 匹配。
- 仅有逻辑 source、无上述对应位置的独立源文件：10 个。正文仍存在于冻结 JSON，但独立官方来源/原文件来源链尚未建立。
- 有意缺少用户事实的有效负例：qa_004 缺用户日志与损坏诊断信息，qa_005 缺本机实际型号/版本。不能给它们虚构支持答案的 Chunk，也不能误记为审核无法判断。

### 6.2 主要风险

1. 标签仍是教学初稿，19 题没有本轮人工签核；16 题没有重点争议不等于 verified。
2. 等级 1 的背景/限制边界不够操作化。q02/q04/q13 应先裁决，避免正例分母和排序指标不稳定。
3. 未标注即 0 使漏标影响指标。未来扩充语料时要复审旧题，不只给新题写标签。
4. 合成正文只能说明语料内部支撑，不能替代官方事实依据、真实日志及版本来源；尤其不要推广示例 CUDA 编号。
5. 18 个短 Chunk 和少量题覆盖有限：api_stream 没有对应检索题；真实噪声、长文档、多约束、跨段、多跳与版本冲突不足。
6. 19 条记录不是 19 个独立意图：qa_001 与 q01/q13、qa_002 与 q02、qa_003 与 q10 高度重合。近似问法/共享证据不能随机分到开发与验收两边后仍声称独立。
7. answerable=false 只指所请求的事实/确定结论缺证，不禁止回答限制说明；未来加入真实环境清单会改变 qa_005 的有效性。
8. 标准答案不等于每句话都是必答项。qa_001 的具体失败原因、qa_003 的候选限制在参考答案中，但未列为独立 required_fact；评分严格度需人工固定。
9. Day 8 文档的没有真实 LLM 输出等表述是当时记录，不作为今天配置/采集状态的结论；本轮不改历史文档。

## 7. 扩充至 50–100 条的覆盖建议（不写入原数据）

可以先规划约 60 个问题意图的新版本；以下只是建议，不是已标注样本。按意图与证据簇分开发/验收集，记录条数和独立意图数分开统计。

| 题型 | 60 条中的参考配额 | 证据与审核重点 |
|---|---:|---|
| 术语与单文档事实 | 12 | RAG/Embedding/FAISS/BM25/Streaming；逐事实出处 |
| 条件、对比与否定 | 10 | cosine 条件、不可预计算、分数尺度；避免机械关键词 |
| 技术故障排查 | 10 | 脱敏真实日志、完整上下文和版本；示例必须明示 |
| 跨文档综合 | 10 | 每个必答事实对应支持 Chunk，互补证据逐项标注 |
| 精确词项/近似编号干扰 | 8 | 型号与相邻编号，区分提及与回答 |
| 证据不足/无法回答 | 10 | 用户环境缺失、版本缺失、事实不在库、冲突；允许限制说明，禁止编造具体值 |

同义改写、口语、错别字、不同问题长度应横跨上述类型，不用重复问法凑独立意图。扩充到 100 条优先补新证据与真实场景。

每条记录建议保留稳定 ID、原始问题、语料版本/指纹、标准答案、逐事实证据 ID/位置、相关性理由、标注人、审核人、日期与裁决说明。无可靠证据或尚未审核的题保持 pending，发现冲突先记录，不用模型补正确答案。任何数据扩充或修订都等待你确认后另建版本，Day 7/8 冻结基线保留。

## 8. 本地验证、文件保护与完成情况

现有 load_dataset/load_qa_dataset 均加载成功：唯一 ID、正文、等级、必填字段、事实清单、文档引用与 SHA 绑定通过。未初始化真实 Embedding/Reranker/LLM。

运行的已有本地测试（未执行默认评估入口）：

    cd D:\RAGrepository
    $env:PYTHONUTF8 = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_retrieval_metrics.py tests/test_retrieval_evaluator.py tests/test_qa_evaluation.py

实际结果：122 passed、0 failed、0 skipped；1 条既有 jieba/pkg_resources 弃用警告。未跑全量模型测试，不宣称重新验证了全部 410 项。测试保证代码约束，不代替人工语义审核。

报告创建前复核：原有 13 个保护文件均与审核前 SHA256 一致。写入本报告后还会再次核对。下表为保护基准，原文件不写入：

| 原有文件 | SHA256（前后须一致） |
|---|---|
| data/evaluation/qa_eval.json | 9e33ac59dd0ba406a047249442672a5a01e22ef73f8f4d3c4eb054133d89c1eb |
| data/evaluation/retrieval_eval.json | 085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71 |
| results/day7_retrieval.json | 8c49a887f05ad146311b84bff3dc83fd952baf18c4e9d059b985dfc4d172e2e6 |
| results/day7_retrieval.txt | b3e02ebf24067f329907cc5a4232779f521f72ba74249614a3f4650126569445 |
| results/day8_qa_collect.txt | 6d79999416374e27e42c7d5c7a2c09968f29e7c3f7bdb707ed9a2cd3c6c2af3b |
| results/day8_qa_deepseek_ai_review_v1.json | 21c22fe324a06d549d1ab10b8edad618e202f5fc60916e1af7770a79ca0ef2e7 |
| results/day8_qa_deepseek_reviewed_summary_v1.json | fad7ae432f75f0d2106f6ec57f1414dc8af2dd3e570aaa0da2ff9e5f2bfae936 |
| results/day8_qa_deepseek_reviewed_v1.before_qa001_2329d1e4.json | 5e912e10ca821b2818b8c762b42b8246e781e9ecfa52a0fb045608e25a444e8c |
| results/day8_qa_deepseek_reviewed_v1.json | 1b95fa47608ddca85d9ca7c45b1d901d45ab645e7dfa00f1f8cd23baa974ea29 |
| results/day8_qa_deepseek_summary_v1.json | 64d10a1a15101d085082cda2c2b7778dbd15b276d832ff8b5b371c8e7df5527e |
| results/day8_qa_deepseek_v1.json | 333899db1b3e45b35335c0b4fed7e282b3181205921ccdd5530a0276af84b57f |
| results/day8_qa_mock.json | 2468464084e9755e9a9de21c65419dea39fe3119627ae012e5b183f8ace6b48d |
| results/day8_qa_summary.json | fa3ec4ccfb3eef4398219f0952e91397a2ebbe5d7b29f2956cba9a1f1bf73ecb |

本轮唯一新增文件：docs/evaluation_dataset_audit.md。没有新增问题或修改数据、代码、Day 7/8 历史报告及人工记录。

完成的是正文核查与独立报告，不是用户人工签核。19 条保持 pending，0 条 verified。等待你审核，优先裁决 q02/q04/q13，再决定是否修订和扩充。本轮到此停止。

## 附录 A：18 个评估 Chunk 的原始正文

以下原样摘录 documents[].text。source 是原元数据，不伪造成存在的独立源文件。QA 负例的闭集排查范围为全部 18 条。

### rag_overview

- ID：rag_overview；source：rag.md；chunk_id：0。
- 原始位置：[rag_overview](D:/RAGrepository/data/evaluation/retrieval_eval.json:14)。

> RAG 是检索增强生成技术。系统先从外部知识库检索与问题相关的信息，再把检索结果放入提示词，交给大语言模型组织答案。

### rag_evidence

- ID：rag_evidence；source：rag.md；chunk_id：1。
- 原始位置：[rag_evidence](D:/RAGrepository/data/evaluation/retrieval_eval.json:15)。

> RAG 可以降低大模型幻觉，因为它向模型提供可参考的外部事实和证据，减少仅凭参数记忆编造答案的情况。模型需要根据检索到的资料作答，资料不足时应说明无法回答。

### rag_limits

- ID：rag_limits；source：rag.md；chunk_id：2。
- 原始位置：[rag_limits](D:/RAGrepository/data/evaluation/retrieval_eval.json:16)。

> RAG 不能保证完全消除幻觉。检索错误、资料过时、上下文缺失或模型误读都可能导致错误回答。应该核查来源和引用，并在知识库缺乏答案时明确说明。

### rag_terms

- ID：rag_terms；source：rag.md；chunk_id：3。
- 原始位置：[rag_terms](D:/RAGrepository/data/evaluation/retrieval_eval.json:17)。

> RAG、大模型、幻觉、检索增强生成是这一页的关键词。这只是术语索引，不解释 RAG 为什么可以降低大模型幻觉，也没有给出相关机制。

### faiss_ip

- ID：faiss_ip；source：faiss.md；chunk_id：0。
- 原始位置：[faiss_ip](D:/RAGrepository/data/evaluation/retrieval_eval.json:18)。

> FAISS IndexFlatIP 对索引内全部文档向量执行精确内积搜索，再返回内积最大的 Top-K。只有查询和文档向量都做单位长度归一化时，内积才等于余弦相似度。

### cosine_unit

- ID：cosine_unit；source：faiss.md；chunk_id：1。
- 原始位置：[cosine_unit](D:/RAGrepository/data/evaluation/retrieval_eval.json:19)。

> L2 归一化把向量除以它的欧几里得长度，使向量范数等于 1。若查询向量 q 和文档向量 d 都已归一化，cosine(q,d)=q·d，因此可用 IndexFlatIP 实现余弦检索。

### embedding_bge

- ID：embedding_bge；source：embedding.md；chunk_id：0。
- 原始位置：[embedding_bge](D:/RAGrepository/data/evaluation/retrieval_eval.json:20)。

> Embedding 将文本编码为稠密语义向量，使含义接近的文本通常在向量空间更接近。本项目使用 sentence-transformers 加载 BAAI/bge-small-zh-v1.5 中文模型，文档向量可以提前计算。

### chunk_overlap

- ID：chunk_overlap；source：chunking.md；chunk_id：0。
- 原始位置：[chunk_overlap](D:/RAGrepository/data/evaluation/retrieval_eval.json:21)。

> 文本切块把长文档拆成较短的 chunk。chunk_overlap 使相邻块保留一段重复内容，减少句子和上下文在边界被截断的影响。overlap 必须小于 chunk_size，否则步长不能正常前进。

### bm25_terms

- ID：bm25_terms；source：retrieval.md；chunk_id：0。
- 原始位置：[bm25_terms](D:/RAGrepository/data/evaluation/retrieval_eval.json:22)。

> BM25 是基于词项匹配的检索方法，综合词频、文档频率和文档长度计算分数。BM25L 和 jieba 分词适合保留错误编号、产品型号等精确关键词；纯向量语义检索可能混淆很相似的编号。

### hybrid_fusion

- ID：hybrid_fusion；source：retrieval.md；chunk_id：1。
- 原始位置：[hybrid_fusion](D:/RAGrepository/data/evaluation/retrieval_eval.json:23)。

> Hybrid Search 融合向量语义匹配和 BM25 关键词匹配。两路原始分数范围不同，应分别做 Min-Max 归一化，再按 alpha*vector_norm+(1-alpha)*bm25_norm 融合。本项目默认 alpha=0.7。

### reranker_pairs

- ID：reranker_pairs；source：retrieval.md；chunk_id：2。
- 原始位置：[reranker_pairs](D:/RAGrepository/data/evaluation/retrieval_eval.json:24)。

> Cross-Encoder 将 query 与 document.text 组成文本对，联合编码后输出相关性分数。同一篇文档面对不同问题会得到不同分数，因此不能提前计算适用于所有问题的固定 rerank score。它只重排检索召回的候选。

### cuda_1175

- ID：cuda_1175；source：cuda_examples.md；chunk_id：0。
- 原始位置：[cuda_1175](D:/RAGrepository/data/evaluation/retrieval_eval.json:25)。

> 这份示例日志中的 CUDA error 1175 后面出现 driver initialization failed。排查时先记录完整错误日志，再核对 GPU 驱动、CUDA 运行时与框架版本。编号本身不足以确定修复方法，不应猜测。

### cuda_1176

- ID：cuda_1176；source：cuda_examples.md；chunk_id：1。
- 原始位置：[cuda_1176](D:/RAGrepository/data/evaluation/retrieval_eval.json:26)。

> 另一份示例日志中的 CUDA error 1176 后面出现 device index out of range。该示例应检查请求的设备序号和可见 GPU 列表，不能把 CUDA error 1175 的日志线索直接套过来。

### cuda_oom

- ID：cuda_oom；source：cuda_examples.md；chunk_id：2。
- 原始位置：[cuda_oom](D:/RAGrepository/data/evaluation/retrieval_eval.json:27)。

> CUDA out of memory 表示当前计算需要的 GPU 显存不足。可先减少 batch size、缩短输入或释放不再使用的张量，并检查是否有其他进程占用显存；这与示例错误编号 1175 的驱动初始化线索不同。

### transformer_attention

- ID：transformer_attention；source：transformer.md；chunk_id：0。
- 原始位置：[transformer_attention](D:/RAGrepository/data/evaluation/retrieval_eval.json:28)。

> Transformer 的 Self-Attention 让每个 Token 根据 Query、Key、Value 与其他 Token 交互，建模上下文关系。注意力权重来自缩放点积及 softmax，而不是简单按词频匹配。

### python_venv

- ID：python_venv；source：setup.md；chunk_id：0。
- 原始位置：[python_venv](D:/RAGrepository/data/evaluation/retrieval_eval.json:29)。

> 在 PowerShell 的项目根目录运行虚拟环境 Python，可使用 .\.venv\Scripts\python.exe main.py。路径开头的 .\ 表示当前目录；项目依赖应安装到同一个 .venv 环境。

### pdf_ocr

- ID：pdf_ocr；source：ingestion_notes.md；chunk_id：0。
- 原始位置：[pdf_ocr](D:/RAGrepository/data/evaluation/retrieval_eval.json:30)。

> 扫描版 PDF 的页面可能只有图片，没有可提取的文本层。普通文本提取返回空字符串时，需要先用 OCR 把图片转成文字，再进行清洗、切块和 Embedding。

### api_stream

- ID：api_stream；source：api.md；chunk_id：0。
- 原始位置：[api_stream](D:/RAGrepository/data/evaluation/retrieval_eval.json:31)。

> FastAPI 可以把 RAG 问答封装为 HTTP 接口。普通响应在答案生成结束后一次返回；StreamingResponse 则可以随生成进度逐段发送内容，让客户端更早看到输出。

## 附录 B：报告写入后的最终保护复核

已在报告写入后重新计算并逐项比较：原有 13 个文件的路径集合与 SHA256 全部一致；两个原始数据集及全部 Day 7/Day 8 历史文件均未变化。报告结构检查确认 14 条检索记录、5 条 QA 记录均为 pending，18 条证据正文与冻结 JSON 原文一致。
