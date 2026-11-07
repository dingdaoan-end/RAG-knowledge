# V3 四方案综合评估实验

日期：2026-10-26（Asia/Shanghai）。实验 ID：9915eade108441c49d79ca173144894d。本报告只使用真实落盘结果；没有根据结果改标签。

## 1. 完成范围与结论边界

- 实验 A：50 个检索问题 × 4 种方案，全部完成。
- 实验 B：200 个不同 QA 单元全部处理；197 个产生真实完整答案，3 个连接失败，无 Mock 填补，no_evidence=0。
- 实际 HTTP 发送尝试 206 次，含 SDK 重试 6 次；197 份成功响应有 usage，记录到 117593 Token。失败请求的服务端用量和计费未知。
- 197 份答案全部留作人工评审，已审核 0。忠实度、切题性、事实正确性、关键事实覆盖与证据不足处理率均未填分，不能宣布总体问答正确率改善。
- 可验证的检索结论：精排提高本数据集的前排排序与 Recall@3，未扩大 Hybrid 的 Top-10 召回集合。问答层有可解释的具体案例，但总体质量结论仍待人工审核。

## 2. 数据、模型与公平条件

| 项目 | 实际配置 |
|---|---|
| 语料 | V3 的 18 个内嵌 Chunk，原文、ID、标签未变 |
| 检索题 / QA 题 | 50 / 50；QA 42 可回答、8 证据不足 |
| 候选 / 最终上下文 | Top-10 / 最多 Top-3；本批 197 份成功答案实际都是 3 个上下文 Chunk |
| Vector | BAAI/bge-small-zh-v1.5，归一化 Embedding，FAISS IndexFlatIP |
| BM25 | 既有 BM25L + jieba；保留实际候选数，不补齐 |
| Hybrid | 既有 Min-Max + alpha=0.7，不改融合公式 |
| Reranker | BAAI/bge-reranker-base，仅重排 Hybrid 实际 Top-10 |
| 生成 | DeepSeek / deepseek-flash / https://api.deepseek.com |
| 生成参数 | 生产 OpenAILLM 只显式发送 model、input；197 份响应实际返回 temperature=1.0、top_p=1.0 |
| Prompt | 四方案共用现有 build_prompt/System 规则；没有把参考答案、required_facts、answerable 或分级标签注入 Prompt |
| 引用规则 | 共用 Context Builder 的 Document/source/chunk_id/eval_id/rank/score；生产 Prompt 没有额外强制引用格式，因此没有为某方案单独加引用指令 |
| 执行设备 | CPU；torch_threads=14；模型从本地固定缓存加载，没有下载 |

LLM 参数未做额外覆盖，没有固定随机种子，故再次请求不保证逐字相同。检索方法改变时，实际上下文和其中的排序/分数元数据也会改变；其余生产流程一致。方法次序按 QA 单元轮换，降低固定顺序偏差，仍不能消除网络波动与服务端缓存效应。

### 指纹与运行环境

- 检索 JSON SHA256：33a7d07e4554df865667592b3515133c210b23678c4fcb895a8cdfa9240e04d9
- QA JSON SHA256：b854564a846d0ca022ffcdea59deb6487cbeab9ecdae828d4a41a749775548c4
- QA corpus_sha256 绑定整个检索 JSON 字节，不是仅正文哈希；现有加载器实际通过。
- BGE 缓存 revision：7999e1d3359715c523056ef9478215996d62a620
- Cross-Encoder 缓存 revision：2cfc18c9415c912f9d8155881c133215df768a70
- Python：3.11.16 (main, Sep  1 2026, 14:15:24) [MSC v.1944 64 bit (AMD64)]
- 依赖版本：numpy=1.26.4, faiss-cpu=1.8.0.post1, sentence-transformers=3.3.1, rank-bm25=0.2.2, jieba=0.42.1, torch=2.14.0, openai=3.29.0, httpx2=2.13.1
- 生产 Prompt 源码 SHA256：12a96aa1195f6534d5e2d43822c3ec5a1ecec7e161273f49aa360061b0ecb1ab
- 原始运行脚本 SHA256：cbd3c1dfc6f635dafe7d373032c313e5d553c282972b3ae4d4d945dfa65b01d5
- 使用 Settings.from_env()；工具原进程未继承配置时，通过 Windows CreateEnvironmentBlock 启动继承既有用户环境的新进程。没有写入或修改环境变量，没有读取 .env。

## 3. 实验 A：真实离线检索指标

复用现有 evaluate_methods 和三个指标模块。相关性 1/2/3 均计二元正例，nDCG 增益为 2^rel-1；排名从 1 开始，跨问题宏平均，MRR 范围统一为 Top-10。

| Method | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@3 | nDCG@5 | nDCG@10 | 平均候选数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector | 0.882000 | 0.927000 | 0.961000 | 0.990000 | 0.961251 | 0.964810 | 0.971228 | 10.00 |
| BM25L | 0.828667 | 0.927000 | 0.975000 | 0.940000 | 0.902416 | 0.925464 | 0.935813 | 9.56 |
| Hybrid | 0.885333 | 0.953667 | 0.967667 | 0.980000 | 0.960652 | 0.970535 | 0.972568 | 10.00 |
| Hybrid + Reranker | 0.911333 | 0.945333 | 0.967667 | 1.000000 | 0.976200 | 0.983482 | 0.987185 | 10.00 |

完整逐题 Top-10 排名、分级标签与原始分数：[检索 JSON](D:/RAGrepository/results/v3_retrieval_comparison.json)；可读输出：[检索 TXT](D:/RAGrepository/results/v3_retrieval_comparison.txt)。

### 召回不足与排序不足分别看

| Method | Recall@10 未满的 Query 数 | 首个相关结果未排第 1 的 Query 数 | 完全没有正相关命中 |
|---|---:|---:|---:|
| Vector | 6 | 1 | 0 |
| BM25L | 4 | 6 | 0 |
| Hybrid | 5 | 2 | 0 |
| Hybrid + Reranker | 5 | 0 | 0 |

Hybrid 的漏召回 Query 为 q02、q29、q33、q38、q41，遗漏的是 embedding_bge、pdf_ocr、embedding_bge、cuda_oom、hybrid_fusion，均为当前标注等级 1。精排没有这些候选，不能补回，所以两者 Recall@10 完全相同。弱相关标注也进入 Recall 分母，不能把这些数值直接称作关键答案事实丢失率。

### 四种方法的优势与失误案例

| 案例 | 实际数据与解释 |
|---|---|
| Vector：q01 / q04 | q01 首位是 rag_evidence，nDCG@3=0.798485，高于 Hybrid 的 0.523434；q04 却把 cuda_1176 排到 cuda_1175 之前，RR=0.5，nDCG@3=0.644287。语义接近不保证编号与诊断线索正确。 |
| BM25：q02、q38 / q47 | q02、q38 的 Recall@10 都为 1；q38 Top-3 三条均为等级 2，nDCG@3=1。但 q47 Top-3 为 pdf_ocr、rag_evidence、rag_limits，只有第一条相关，Recall@3=1/6，nDCG@3=0.469279。精确词项匹配并不自动保证多文档组合完整。 |
| Hybrid：q47 / q01 | q47 Recall@3=0.5，优于 BM25 的 1/6；q01 却受关键词噪声影响，将明确不解释机制的 rag_terms 排首，nDCG@3 低于 Vector。Min-Max 解决分数尺度，无法自动消除错误信号。 |
| 精排：q04 / q38 | q04 调整为 cuda_1175、cuda_oom、cuda_1176，RR=1、nDCG@3=1；q38 却将不相关 reranker_pairs 放入第三位，使 Recall@3 从 0.6 降至 0.4、nDCG@3 从 1 降至 0.765361。MRR=1 不等于整份排序都正确。 |

精排相对 Hybrid：Recall@3 提高 2.60 个百分点，MRR@10 提高 2.00 个百分点，nDCG@3 提高约 1.55 个百分点；Recall@10 不变。Recall@5 从 0.953667 降至 0.945333：q24 改善，q28/q33/q47 的部分弱相关背景移出前五。未为改善数值修改标注。

## 4. 实验 B：调用实况、中断与恢复

最初 qa_001、qa_004 的四方案共 8 个正式单元构成 Smoke Test，均成功并直接纳入结果，未重复请求。随后继续批次。初批完成 174 单元时，出现三个连续 APIConnectionError，SDK 各重试 2 次后停止。

| Method | 单元 | 成功答案 | 失败单元 | no_evidence | HTTP 尝试 | SDK 重试 | usage 可用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector | 50 | 49 | 1 | 0 | 52 | 2 | 49 |
| BM25L | 50 | 49 | 1 | 0 | 52 | 2 | 49 |
| Hybrid | 50 | 50 | 0 | 0 | 50 | 0 | 50 |
| Hybrid + Reranker | 50 | 49 | 1 | 0 | 52 | 2 | 49 |

失败单元为 bm25__qa_043、hybrid_reranker__qa_044、vector__qa_044，失败阶段均为 LLM，异常类型均为 APIConnectionError；没有保存原始异常请求、堆栈或密钥。

其中 bm25__qa_043 的首次尝试收到 HTTP 200，但随后响应未完整读取，最终没有答案或 usage；其单元总耗时约 701.124 秒。收到 200 响应头不等于成功生成。记录不足以确定长间隔由主机暂停、网络、代理或上游哪一部分造成，不能据此诊断为 BM25 慢，也不能认定服务端未处理或未计费。

在不发送额外模型请求的 TCP443 检查恢复后，只续跑 26 个从未开始的单元，全部成功；原有 174 单元连同 3 个失败单元全部复用，失败请求没有再重放。完成后共有 200 个不同单元、197 份完整答案；206 次 HTTP 尝试中，198 次有 200 响应头，8 次没有响应头，其中一份 200 未完整生成，故有效答案仍为 197。

中止时的 JSON/TXT 和四份人工评审输入原字节保存在 [中止快照目录](D:/RAGrepository/results/v3_interrupted_snapshot)，随后仅接替本轮活动汇总。初批与续跑均保留相同 Provider、模型、Prompt 和数据。补跑的初始化/时段单独记录，没有隐藏网络异常或将失败改成 Mock 成功。

时间信息：实验记录开始 2026-10-26 21:46:51；续跑 2026-10-26 22:19:13 至 2026-10-26 22:20:17（北京时间）。这段墙钟时间包括中断和恢复处理，不等于模型推理时间。

## 5. 真实答案与上下文案例（不是人工评分）

### 5.1 排序改善不必然改变已经具备证据的答案

检索 q01 的 Hybrid 将术语索引排首、精排将机制文档排首；对应主题的 QA qa_001 在四种方法的 Top-3 都包含 rag_evidence 与 rag_limits，四个实际答案都表达了外部证据和不能完全消除幻觉。检索题与 QA 题文本不同，这里是主题案例对照，不把 q01 的指标当作 qa_001 的逐题分数。

例如 [hybrid / qa_001](D:/RAGrepository/results/v3_qa_progress/hybrid__qa_001.json) 写道：“RAG 不能保证完全消除幻觉。” [hybrid_reranker / qa_001](D:/RAGrepository/results/v3_qa_progress/hybrid_reranker__qa_001.json) 同样写出该结论。该案例支持“排序变好后答案未必发生明显事实变化”的观察，不构成总体质量相同的统计证明。

qa_003 中 Hybrid/Vector 的上下文包含无关 CUDA/切块材料，精排改为 reranker_pairs、embedding_bge、faiss_ip，证据更贴近问题；四份答案仍都解释了 query-document 联合编码。不能凭背景替换就宣布事实正确性提高。

### 5.2 排序截断造成的实际漏答：qa_031

问题询问 OCR 与 Embedding 各自输入输出。BM25 Top-10 含 embedding_bge，但它排第 6，最终上下文只有 pdf_ocr、hybrid_fusion、api_stream。模型实际回答：

> 资料只说明文本经过清洗、切块后进入 Embedding 流程，但没有明确说明它把什么转换成什么。

见 [bm25 / qa_031](D:/RAGrepository/results/v3_qa_progress/bm25__qa_031.json)。Hybrid 的上下文包含 pdf_ocr、embedding_bge，答案实际写出“文本编码成稠密语义向量”；精排也将这两条资料交给模型。见 [hybrid / qa_031](D:/RAGrepository/results/v3_qa_progress/hybrid__qa_031.json)、[hybrid_reranker / qa_031](D:/RAGrepository/results/v3_qa_progress/hybrid_reranker__qa_031.json)。

这是候选已召回、Top-3 排序/截断未提供关键证据的可定位问题。BM25 的局部拒答可能忠实于它实际收到的上下文，同时没有完整回答原问题；忠实度与切题/事实覆盖必须分开人工审核。不能把任何“无法回答”都称作可靠性成功。

### 5.3 多文档问题与预算瓶颈：qa_040

本题需要切块、文本向量化、内积搜索、RAG 提示词四方面信息。四种方法的 Top-10 均包含 chunk_overlap、embedding_bge、faiss_ip、rag_overview；实际 Top-3 却未覆盖完整链路。Hybrid 与精排最终都缺少前两条，答案都明确表示无法完整说明。

参见 [hybrid / qa_040](D:/RAGrepository/results/v3_qa_progress/hybrid__qa_040.json)、[hybrid_reranker / qa_040](D:/RAGrepository/results/v3_qa_progress/hybrid_reranker__qa_040.json)。精排在此没有补齐问题所需信息；Top-3 数量预算与单 Chunk 内容范围共同限制证据覆盖。不能仅凭全局 MRR@10=1 判断最终 Prompt 足够。

### 5.4 同样证据也可能有生成解释差异：qa_043

成功的 Vector/Hybrid/精排均收到 transformer_attention、faiss_ip、cosine_unit 这三条资料（顺序不同）。Hybrid 文字中出现“资料明确提到输出‘注意力权重’”的表述，Vector/精排则对 Self-Attention 最终输出形式保持限制说明。

源正文只描述 Token 交互和权重来自缩放点积/softmax，并未完整定义最终输出。该措辞是否构成过度归纳，应由人工对忠实度和事实正确性分别复核；本轮未自动认定错误或填 unsupported_claims。参见 [hybrid / qa_043](D:/RAGrepository/results/v3_qa_progress/hybrid__qa_043.json)、[vector / qa_043](D:/RAGrepository/results/v3_qa_progress/vector__qa_043.json)。

## 6. 八道证据不足题：32 份实际回答

以下是逐条阅读后的文字观察，不是拒答率、幻觉率或人工评分。32 份均有非空检索上下文，Pipeline 状态都是 answered，no_evidence 为 0；answered 只表示真实生成路径完成，不能解释成证据充分或答案正确。

| QA ID | 所请求但缺少的事实 | 四种方案的实际回答观察 |
|---|---|---|
| qa_004 | 显卡损坏结论与唯一修复命令 | 四份均否定仅凭编号确定，引用示例日志/需要完整日志与版本；没有给出唯一命令 |
| qa_005 | 本机 GPU 型号、驱动、CUDA 版本 | 四份均说明资料没有实际值；均未明确请求用户提供当前环境检测结果，这一必答项可能缺失，待人工判定 |
| qa_045 | alpha=0.7 最优的实验证明 | 四份均说明只给默认值、缺不同权重对照；没有把默认值宣称为最优 |
| qa_046 | 本机十万向量检索毫秒数 | 四份均不提供具体毫秒值；精排答案明确提出需要实际测试，其余补证建议详略不同 |
| qa_047 | 当前 PDF 的 OCR 准确率与全对保证 | 四份均不给准确率或全对保证；精排提出核查原 PDF/OCR 结果。其他答案引用 RAG 错误限制时，是否恰当区分 OCR 与生成误差仍应复核 |
| qa_048 | 本次流式响应节省的毫秒数 | 四份均说明只有定性发送说明，缺少本次计时数据 |
| qa_049 | 本机 GPU 数量与正确设备号 | 四份均拒绝根据示例填写数量或设备号；精排明确提出进一步查看可见 GPU 列表 |
| qa_050 | 某句话的具体 Embedding 向量数值 | 四份均说明仅有模型/方法介绍，没有实际编码数值，未编造具体向量 |

原始回答可按 method__qa_id.json 在 [逐单元目录](D:/RAGrepository/results/v3_qa_progress) 定位，亦完整汇总在 QA JSON/TXT。上述样例显示模型能够表达限制，但无法据此判断哪种方案总体更不容易产生无依据断言；197 份人工评审均未完成，补证请求和关键事实覆盖仍需逐项检查。

## 7. 性能与 Token：完整分布和公平子集

### 7.1 全部 50 单元/方法，保留失败与异常

| Method | 平均检索 ms | 平均重排 ms | 平均 LLM s | 总耗时均值 s | 中位数 s | P95 s | 最大 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector | 11.486 | - | 2.296 | 2.331 | 1.915 | 4.400 | 10.238 |
| BM25L | 0.544 | - | 16.112 | 16.132 | 1.895 | 3.850 | 701.124 |
| Hybrid | 12.576 | - | 2.152 | 2.187 | 2.011 | 3.687 | 5.654 |
| Hybrid + Reranker | 15.084 | 501.698 | 2.347 | 2.888 | 2.364 | 5.564 | 11.405 |

BM25 的约 701 秒失败单元拉高完整均值。失败在连接异常时段集中发生；本批 Hybrid 恰好 50/50 成功不能证明其检索实现更能抵抗网络故障。完整分布已保留，没有删除慢请求美化结果。

### 7.2 四方案均成功的同一 48 道 QA 对照

排除 qa_043、qa_044 整道问题的四方案记录，再对剩下相同 48 个 QA ID 分别统计。并非仅删除 BM25 的慢项。这里是成功请求条件下的延迟对照，不是新的答案质量指标。

| Method | 检索均值 ms | 重排均值 ms | LLM 均值 s | 总均值 s | 总中位数 s | 总 P95 s |
|---|---:|---:|---:|---:|---:|---:|
| Vector | 11.604 | - | 2.085 | 2.119 | 1.901 | 3.735 |
| BM25L | 0.448 | - | 2.102 | 2.120 | 1.885 | 3.711 |
| Hybrid | 11.509 | - | 2.062 | 2.096 | 1.992 | 3.358 |
| Hybrid + Reranker | 11.533 | 494.964 | 2.119 | 2.649 | 2.337 | 4.451 |

在此子集中，精排平均额外执行约 495 ms 的 Cross-Encoder，端到端均值约 2.649 s，Hybrid 约 2.096 s。BM25 本地检索约 0.448 ms，明显小于 Vector 的约 11.604 ms，但 LLM/网络占主要部分，端到端并未按本地检索差异等比例下降。这些是本次单机和云服务时段的观察，不作统计显著性或部署 SLA 结论。

### 7.3 用量与费用边界

| Method | usage 记录数 | 输入 Token | 输出 Token | 总 Token | 缓存输入 Token（已包含在输入中） |
|---|---:|---:|---:|---:|---:|
| Vector | 49 | 18179 | 10878 | 29057 | 0 |
| BM25L | 49 | 18095 | 10958 | 29053 | 0 |
| Hybrid | 50 | 18377 | 11752 | 30129 | 0 |
| Hybrid + Reranker | 49 | 18155 | 11199 | 29354 | 767 |

可见合计：输入 72806、输出 44787、总计 117593 Token，缓存输入 767 不额外叠加。只覆盖 197 个有 usage 的完整响应；3 个失败单元及重试的服务器用量未知，117593 不是已核对的全账单 Token。没有可信的逐请求账单及价格快照，因此费用保持 null，不估算金额。

服务端缓存计数和输出长度因上下文与生成随机性变化；不能只按总 Token 给方法排名。相同 48 题的总 Token 分别为 Vector 28068、BM25 28262、Hybrid 28189、精排 28275，差异很小，也不代表答案质量差异或可靠的费用差异。

### 7.4 冷启动和计时定义

- 初批进程初始化约 7.266 s；续跑进程初始化约 17.393 s。分别含本地模型加载、索引准备、模块导入等，没有模型下载。
- 初始化不计入每个 QA 单元的检索/重排/总耗时。实验 A 也让模型完成了实际推理，随后 B 主要为已加载模型请求。
- 检索计时包括该方法的查询编码与实际搜索；重排计时为候选打分及排序。LLM 计时包括 SDK、HTTP、有限重试及必要进度落盘，不是纯模型推理时间。
- 总耗时包括生产 Pipeline 与 Day 8 采集包装，不含最终汇总写盘和模型初始化；没有额外启动 FastAPI，因此不是浏览器到服务端的 HTTP 全链路 SLA。
- 原始 JSON 另有每方案排除首个单元的统计；SDK 初次连接/TLS未单独计时。网络失败、主机暂停和续跑时段等影响无法完全分解，不能把 701 秒解释为某个检索算法计算时间。

## 8. 检索和问答的联合解释

1. Recall 更高不等于答案更好。BM25 的 Recall@10 最高，但 qa_031 的 embedding_bge 虽在候选中却没进入 Prompt；它的答案仍未解释完整。最终预算和排序同样重要。
2. 精排改善排序有数字依据（q04、全局 nDCG/MRR），但目前没有人工事实正确性均值，不能宣称最终准确率随之提高。q01/qa_001 也说明关键证据已经在 Top-3 时，排序变化未必改变核心回答。
3. 相近的全局指标会掩盖局部上下文差异；qa_031、qa_040 分别表现为单个关键来源被截断与多文档链路缺证。
4. 召回不足看 q02/q29/q33/q38/q41 的候选缺失；排序/截断看 qa_031；最终预算看 qa_040；生成解释待复核看 qa_043。实际网络失败另列，不能归入检索或事实性错误。
5. Hybrid 会受关键词噪声影响，q01 将 rag_terms 提到首位；这批 qa_001 仍因其余上下文给出机制和限制说明，噪声不必然造成这次错误答案。
6. 精排在 qa_003 引入更贴近问题的 Embedding 背景，但 qa_040 未补齐流程资料；A 的 q38 还发生局部退化。不能只展示成功案例。
7. 八道负例均有文档却缺所请求的事实；no_evidence 没有触发。当前只有32份具体文字观察，没有人工幻觉率/证据充分性评分，不能比较哪个方案更容易无依据回答。
8. 已知权衡是检索排序、延迟和可见 Token：精排排序领先且增加约半秒本地处理，Vector 的 nDCG@3 已接近 Hybrid，BM25 本地检索便宜但排序弱。总体答案质量/费用谁最平衡仍须人工审核和账单依据。

检索题 qXX 与 QA 题 qa_XXX 是两个问题集合，本轮没有问题文本完全相同的直接对应。记录中的 query_id 是 QA 请求 ID，等于 qa_id；retrieval_eval_query_ids 为空，没有伪造一对一映射。因此本轮只做同主题案例解释，不计算检索分数与人工答案质量的逐题相关系数。

Hybrid 与精排共同有答案的49个QA中，22个使用相同上下文集合，12个连顺序也相同；其他差异中仍可能保留相同关键证据。这是上下文诊断，不能代替答案质量评分。

## 9. 人工评审、结果文件与复现

四份输入均可由现有 Day 8 load_qa_results/summarize_reviews 直接处理：

- [vector：49份待评审答案](D:/RAGrepository/results/v3_qa_review/vector.json)。
- [bm25：49份待评审答案](D:/RAGrepository/results/v3_qa_review/bm25.json)。
- [hybrid：50份待评审答案](D:/RAGrepository/results/v3_qa_review/hybrid.json)。
- [hybrid_reranker：49份待评审答案](D:/RAGrepository/results/v3_qa_review/hybrid_reranker.json)。

3 个失败单元没有伪造答案，不生成可以打分的记录。faithfulness、answer_relevance、factual_correctness、required_facts_covered、unsupported_claims 与 insufficient_evidence_handled 均保持初始待评审状态；最后一项用于 Evidence Sufficiency/负例限制处理。旧 Day 8 的人工分数没有迁入本轮。

建议人工只编辑评审副本的 review，再用现有汇总入口生成新的文件名。例如（本轮未执行评分或汇总成质量分数）：

    .\.venv\Scripts\python.exe scripts/evaluate_qa.py summarize --input results/v3_qa_review/vector.json --output results/v3_qa_review/vector_summary.json

### 结果与审计文件

- [检索 JSON](D:/RAGrepository/results/v3_retrieval_comparison.json) / [检索 TXT](D:/RAGrepository/results/v3_retrieval_comparison.txt)
- [真实 QA JSON](D:/RAGrepository/results/v3_qa_comparison.json) / [真实 QA TXT](D:/RAGrepository/results/v3_qa_comparison.txt)
- [200 个单元 checkpoint：完整答案、实际 Prompt、来源、候选、分数、Token、时间和请求尝试](D:/RAGrepository/results/v3_qa_progress)
- [配置与模型初始化清单](D:/RAGrepository/results/v3_experiment_manifest.json) / [原始文件 SHA256 与运行库版本](D:/RAGrepository/results/v3_input_manifest.json)
- [落盘后独立校验结果](D:/RAGrepository/results/v3_validation.json) / [中止快照](D:/RAGrepository/results/v3_interrupted_snapshot)
- [原始独立实验入口](D:/RAGrepository/scripts/run_v3_comprehensive.py) / [只执行未开始单元的恢复入口](D:/RAGrepository/scripts/continue_v3_unstarted.py)
- [17 项实验流程测试](D:/RAGrepository/tests/test_v3_comprehensive.py) / [5 项恢复测试](D:/RAGrepository/tests/test_v3_continuation.py)
- [续跑时实际代码快照](D:/RAGrepository/results/v3_code_snapshot/continue_v3_unstarted.executed.py)，SHA256=2acc5eabda2386d9a5f6e64b53fa691d1c5d0c013e2b68a1bb934f1f929e43ec。采集后仅在当前恢复入口补充 Prompt 前置指纹与备份字节校验，原执行快照及本批数据不变，没有新增真实请求。

### 实际运行方式

初次运行使用现有配置，不更换 Provider：

    .\.venv\Scripts\python.exe -B -X utf8 scripts/run_v3_comprehensive.py --user-environment

模型都使用 local_files_only 的缓存定位；没有改环境变量来控制下载。Windows 参数只启动继承现有用户环境的子进程；在配置已生效的普通终端中可直接运行脚本。

完成后实际验证过以下命令，返回 complete=true、No calls made，checkpoint 请求计数仍为206：

    .\.venv\Scripts\python.exe -B -X utf8 scripts/run_v3_comprehensive.py --user-environment --resume

本次断网后的未开始单元由独立恢复入口执行；旧成功和失败单元都不会重放。恢复后的首个新单元作为连通性验证，若该单元失败立即停，之后沿用有限 SDK 重试与持续失败停止策略。未知 in_flight 状态必须先人工核对，不能自动再次计费。

复现意味着能核对相同数据、模型 revision、生产源码与调用参数；云端 temperature=1 且无固定种子，不承诺重复运行得到逐字一致的答案。当前路径拒绝覆盖，重新采样应使用独立输出副本保留本轮结果，不能删除历史数据后冒充同一实验。

## 10. 测试、保护与面试可用表述

实际执行：

    .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short -m "not integration"

结果：485 passed、0 failed、0 skipped，10 deselected，1 条既有 jieba/pkg_resources 弃用警告，用时35.31秒。10项为未选择的真实模型集成测试；实际实验另行加载了真实缓存模型。新增流程测试17项、恢复测试5项，均通过。离线 MockTransport 的模拟请求不计入206次真实 HTTP 尝试。

落盘后另行校验：197份实际 Prompt/上下文/来源逐字可重建，候选不超过10、实际上下文不超过3、精排只处理已召回候选，全部200单元唯一且有状态；所有已存在的94个非密钥、非缓存文件 SHA256 一致，其中17个 V1/V2/V3 评测数据与 Day7/8历史文件完全不变。生产算法、指标、默认入口与依赖均未修改。

可用于面试的事实表述：在18个短Chunk和50条检索题的固定基线上比较四方案，精排相对Hybrid将nDCG@3从0.960652提高到0.976200、Recall@3从0.885333提高到0.911333，同时保留Top-10召回上限；构建200格真实QA实验与可续跑日志，197格成功、3格网络失败，所有原始证据、答案、Token和延迟可追溯。不能把这些数字写成生产问答准确率或“幻觉降低百分比”。

局限：语料仅18条合成短文，问题间共享证据；人工质量评审未完成；LLM有随机性与网络/缓存波动，运行包含中断和恢复；价格与失败请求账单未知。尚不能得出生产泛化、显著性、总体幻觉率或最优融合权重的结论。

本轮实验与报告到此停止，不进行调参、改标、额外付费重跑或下一阶段开发。
