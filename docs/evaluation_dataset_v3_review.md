# V3 评测集扩充与质量复核

日期：2026-10-26（Asia/Shanghai）。仅本地数据工程与源码证据核查；未进行真实检索对比实验。

## 1. 交付规模与版本

| 数据 | 基线 | V3 | 保留 / 新增 |
|---|---|---|---|
| 检索集 | data/evaluation/retrieval_eval_v2.json | data/evaluation/retrieval_eval_v3.json | 14 + 36 = 50 题 |
| QA 集 | data/evaluation/qa_eval_v2.json | data/evaluation/qa_eval_v3.json | 5 + 45 = 50 题 |

沿用全部 18 个 Chunk 的原文、稳定 ID、source、chunk_id。检索 q01–q14、QA qa_001–qa_005 对象与 V2 完全一致，q02/embedding_bge=1、q04/cuda_oom=1、q13/rag_overview=1 均保留。新增 ID 为 q15–q50、qa_006–qa_050。

QA 共 42 个可回答样本、8 个无法确定所请求事实的负例，包含 148 项 required_facts。检索正相关证据合计覆盖全部 18 个 Chunk。题目按单一主要题型归类，以下只用于报告，不写入数据字段。

### 检索题型

| 主要题型 | 数量 |
|---|---:|
| 工作原理 | 11 |
| 事实查询 | 9 |
| 故障排查 | 7 |
| 关键词与语义 | 3 |
| 技术比较 | 8 |
| 证据限制 | 3 |
| 概念解释 | 2 |
| 相似概念辨析 | 3 |
| 多文档综合 | 4 |

### QA 题型

| 主要题型 | 数量 |
|---|---:|
| 原理机制 | 11 |
| 参数配置 | 5 |
| 组件比较 | 10 |
| 证据不足 | 10 |
| 基础概念 | 3 |
| 故障排查 | 6 |
| 跨文档综合 | 5 |

QA 的“证据不足”题型包含 8 个 answerable=false 负例，以及 2 个可依据正文回答如何处理证据限制的样本，不可将题型数直接当负例数。

现有检索指标拒绝没有正相关标注的题目，因此检索集的限制类问题检索“不能确定/资料不足应说明”的证据，未强塞零正例问题或修改指标。

## 2. 审核方法、发现与修正

先读取全部 18 条正文及 dataset.py、qa_dataset.py、retrieval_metrics.py、evaluator.py，再逐题检查两份 50 题初稿的语义、引用、等级与答案。进行了第二名 Codex 审查代理的独立只读复核，按实际正文落实以下修正，并复查最终改动；没有把它称为人工认证，也未参考历史实验成绩调标。

| 发现 | 交付前修正 |
|---|---|
| q22 与 q50 都只问流式返回时机，意图区分不足 | 保留 q50；q22 替换为 OCR 文本提取与 HTTP 逐段发送的组件辨析，pdf_ocr/api_stream 各 2 |
| q28 的 pdf_ocr 本身直接给出先 OCR、后切块的顺序，2 分偏低 | pdf_ocr 调为 3，chunk_overlap 仍为 2 |
| q29 漏掉明确描述 OCR 后清洗/切块/Embedding 顺序的背景 | 增加 pdf_ocr:1 |
| q42 的 rag_evidence 不说明具体提示词形式或由谁组织答案，不宜给 2 | 调为背景等级 1，rag_overview 保持直接答案等级 3 |
| q47 的搜索阶段可由单位向量与 IndexFlatIP 的解释提供背景 | 增加 cosine_unit:1 |
| qa_030 对切块步骤表述可更明确，必答项又遗漏了题目要求的边界收益 | 答案明确切块步骤，增加“减少句子和上下文在边界被截断的影响” |
| qa_034 虽问如何启动，但必答项未要求实际命令 | 答案直接给标准 PowerShell 命令，增加命令必答项 |
| qa_038 虽问评分依据，但必答项遗漏 Hybrid 归一化与加权 | 答案及 required_facts 补入 Min-Max 和现有融合公式 |

修正仅涉及本轮新增题。全部问题做了去标点/大小写归一化后的文本去重，并用字符二元组相似度筛查候选，再按意图核对；最终没有完全重复问题。不同子问题仍可能共享证据，不能据此声称 50 个独立知识主题。

全部引用都能通过现有 Loader 定位到 documents[].text。39 个新增可回答 QA 的 115 项关键事实有原文短语可定位；路径证据只把冻结正文展示用的双反斜杠规范为标准单反斜杠，没有改变原 Chunk。原有 5 个 QA 按完整正文逐项复核并保持原样。每个负例则核对全语料确实没有所请求的用户环境、具体向量或实测数值；不虚构支持具体数值的文档。

新负例覆盖默认权重不能证明最优、未测量的检索耗时、未核验的 OCR 准确率、未测量的流式延迟、未知 GPU 清单和未计算的具体向量。已有 hardware/唯一修复命令负例完整保留。

V3 沿用原有记录字段，无 pending、verified、review_status 或新增审核标记；也不携带 V2 元数据中的 annotation_status/annotation_process。只保留评测用途元数据并更新版本、绑定与规模限制。原 V2 元数据不变。

## 3. 指纹、配对与历史保护

检索版本为 metadata.version=3，QA 为 version=3.0。现有指纹算法是整个检索 JSON 原始字节的 SHA256，包含标签、元数据、空格和换行；不是仅对文档正文计算。

| 文件 | 最终 SHA256 |
|---|---|
| retrieval_eval_v3.json | 33a7d07e4554df865667592b3515133c210b23678c4fcb895a8cdfa9240e04d9 |
| qa_eval_v3.json | b854564a846d0ca022ffcdea59deb6487cbeab9ecdae828d4a41a749775548c4 |

QA 的 corpus_sha256 为上表检索指纹。每次修正检索 V3 后都重算 QA 绑定；现有加载器实际验证 V1/V2/V3 正确配对、6 种跨版本错误配对及人为错误指纹，未关闭 SHA 校验或改变默认入口路径。

本轮开始记录 90 个非密钥、非缓存原有文件的 SHA256，包括全部 V1/V2 数据及 Day 7/8 历史文件；最终文件快照核对结果见下节。

## 4. 验证结果

新增 tests/test_evaluation_dataset_v3.py：36 项测试，36 通过、0 失败、0 跳过。相关数据/指标/评估测试：185 通过、0 失败、0 跳过，1 条既有 jieba/pkg_resources 弃用警告。

在 D:/RAGrepository 中设置 PYTHONUTF8=1、PYTHONDONTWRITEBYTECODE=1、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、RUN_MODEL_TESTS=0 后执行：

    .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_retrieval_metrics.py tests/test_retrieval_evaluator.py tests/test_qa_evaluation.py tests/test_evaluation_dataset_v2.py tests/test_evaluation_dataset_v3.py

纯数据测试检查 50/50 数量、顺序保留、唯一 ID、严格字段集合、文档引用及等级、来源可定位、审核修正点、版本配对、错误指纹与历史哈希。现有公式测试及 fake 评估器测试继续复用；没有把单元测试的 fake 组件当成真实四方案实验。

同一离线环境运行全量命令：

    .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short

实际结果：473 项中 463 通过、0 失败、10 跳过，1 条既有 jieba 弃用警告；用时 60.25 秒。10 个跳过项都是由已有 RUN_MODEL_TESTS 开关禁用的真实 BGE 模型集成测试，没有删除测试或改校验。上述测试范围互相包含，通过数不能相加。

测试后再次确认：新增可回答样本的 115 项关键事实不仅可在来源正文中定位，也全部包含在其对应参考答案中。源文件与答案同时可定位不等于自动证明了所有语义判断，语义复核与源文本边界检查仍单独进行。

修改前后快照对比：原有 90 个文件的 SHA256 全部一致，0 个修改、0 个删除；其中 15 个 V1/V2 数据和 Day 7/8 历史文件也全部保持原字节。只新增两份 V3 JSON、本报告和一份 V3 测试文件，生产 app/、scripts/、已有测试、配置及依赖无变化。工作区原有文件尚未跟踪，因此使用完整文件 SHA256 快照与 JSON 对象对比核查范围，而非以空的默认 git diff 作为证明。

## 5. 语料限制与停止范围

只有 18 条短小合成技术正文。扩充的是问题覆盖和组合，不是新增 50 个独立真实场景；检索与 QA 主题重叠，也不构成训练/验收独立划分。source 名称主要为逻辑标签，实际证据在 JSON 内嵌正文，尚无完整外部权威来源链。CUDA 1175/1176 为教学示例，不能当官方编号定义或当前机器的诊断事实。

文字匹配测试保证可定位，无法单独证明语义正确性或穷尽所有弱相关边界；最终标签仍有判断空间，自动审核不等于多人独立人工标注。QA 负例的成立范围是当前冻结语料；未来若引入实测环境或运行结果，须重新审核。

没有修改检索算法、Pipeline、Context Builder、LLM Provider、指标或默认评估入口；没有运行四方案完整对比实验、真实 LLM API、真实模型加载/下载或密钥读取。本轮完成后停止，不进行实验、调参或进一步扩充。
