# 评测集 V2 变更记录与完整性验证

日期：2026-10-26（Asia/Shanghai）。本版只修订独立数据文件，不运行四方案评估。

## 1. 目的与裁决来源

唯一标签修改依据是用户在本次任务中明确提供的三项人工裁决。Codex 负责落实和校验，不将自动语义判断标为人工审核。V2 的 18 个 Chunk、14 个 Query、文档与问题原文、稳定 ID、source/chunk_id 不变。

| Query / Document | V1 | 用户裁决 / V2 | 处理与原因 |
|---|---|---|---|
| q02 / embedding_bge | 1 | 1 | **确认保留，无标签变更**。Embedding 通用背景有一定帮助，但未直接解释单位归一化及内积/余弦条件。 |
| q04 / cuda_oom | 显式 0 | 显式 1 | 能区别 OOM 与 1175 的驱动初始化线索，但不能直接给出完整排查步骤。 |
| q13 / rag_overview | 未列出，按现有规则为 0 | 新增显式 1 | RAG 检索与提示词流程提供弱相关背景，不能直接回答是否完全消除幻觉。 |

全部实际标签差异只有后两行：一个原标签值变化，一个新增标签项。没有删除或调整其他标签；q02 的整个 Query 保持不变。显式标签数由 26 变为 27，正相关标签由 22 变为 24，显式 0 由 4 变为 3。相关性定义、正相关阈值 1、未列出按 0 的规则不变。

## 2. 数据文件、版本及实际 SHA256

唯一检索基线是 data/evaluation/retrieval_eval.json；没有使用旧 data/retrieval_eval.json。

| 路径（相对于 D:/RAGrepository） | 版本 | SHA256 |
|---|---|---|
| data/evaluation/retrieval_eval.json | metadata.version=1 | 085def806d480f92e1a36e832c8fbfcd10a9046a5651b42eef13df68b5fb9d71 |
| data/evaluation/retrieval_eval_v2.json | metadata.version=2 | d204aef8d060a86909f8a239e8bf93630ce375136db9efbf971c28539ecc458e |
| data/evaluation/qa_eval.json | version=1.0 | 9e33ac59dd0ba406a047249442672a5a01e22ef73f8f4d3c4eb054133d89c1eb |
| data/evaluation/qa_eval_v2.json | version=2.0 | fa24d5ca860ecf2c07365d662f5ff1cc5da00a2e86e9e67c5189c3a4752f3092 |

检索元数据仅更新现有的 name、version、annotation_status、annotation_process。状态为 teaching_labels_partially_human_adjudicated_remaining_pending，process 明确只有 q02/embedding_bge、q04/cuda_oom、q13/rag_overview 三项已由用户裁决，其他历史标签继续待人工复核。没有新增标签 Schema，也没有把全库标为 verified。

QA 仅更新 version、corpus_sha256、metadata.name、metadata.corpus_file。annotation_status、全部 5 条 samples（问题、参考答案、required_facts、文档引用、answerable）及其余元数据保持原样。QA 版本变化只表示配套绑定修订，不表示答案或标签获得新增人工认证。

## 3. corpus_sha256 实际算法及兼容处理

实际实现属于任务中的**情况 A**：

- app/evaluation/dataset.py 的 load_dataset 读取 Path(path).read_bytes()，返回 hashlib.sha256(content).hexdigest()。
- 输入是整个检索 JSON 文件的原始字节，包含 documents、queries、metadata、空格与换行；不是仅对语料正文或 ID 的规范化哈希。
- app/evaluation/qa_dataset.py 比较 QA 的 corpus_sha256 与传入 EvaluationDataset.sha256，不相等就抛 ValueError。QA 自身也有独立的文件字节 SHA256。
- 本次虽未改变文档正文，标签和版本元数据的变化仍使检索指纹从上述 V1 值变为 V2 值。因此必须新建 qa_eval_v2.json，并将 corpus_sha256 设为 V2 检索指纹。
- metadata.corpus_file 是描述信息；实际加载仍由调用者传入语料对象，不能只改文件名绕过 SHA 校验。
- 原 QA + 原检索集继续合法。V2 QA + V2 检索集应通过现有加载器；新旧交叉配对应拒绝。这种拒绝是保护行为，不是放宽校验的理由。
- 没有修改任何加载器、默认路径或脚本行为；将来运行 V2 时需显式指定两个配套文件，本轮没有运行评估入口。

## 4. 原始文件保护

开始时记录了 86 个非密钥、非缓存项目文件的 SHA256，其中 13 个为原始评测数据与 Day 7/8 历史文件。没有读取 .env 文件或密钥。原始保护指纹保存在新增回归测试的常量中，最终还会与本轮开始时的完整文件快照比较。

重点保护：两个原始 JSON、Day 7 JSON/TXT、Day 8 Mock/真实采集、人工审核工作副本、审核前备份、AI 独立审核及历史汇总。此前的 evaluation_dataset_audit.md 也只作为历史报告保留，不改写其 pending 记录；本次用户裁决另记在本文件及 V2 元数据中。

## 5. 验证记录

以下均为实际执行结果，不包含真实模型推理或真实 API 请求。

### 5.1 实际数据加载与差异校验

- 现有 load_dataset 分别成功加载 V1/V2；两者都是 18 个 Chunk、14 个 Query，Document 正文及 metadata 完全相同。
- 现有 load_qa_dataset 成功加载“原 QA + 原检索”和“V2 QA + V2 检索”；两个交叉配对都抛出预期的语料 SHA256 不一致错误。
- 对两个版本全部标签作结构比较，实际变化恰为 q04/cuda_oom:0→1、q13/rag_overview:未列出→1；q02 的完整 Query 对象不变。
- 5 条 QA samples 完全相等；V2 的实际字节哈希与第 2 节一致，没有修改标准答案、事实清单或负例标签。

### 5.2 测试命令与结果

在 D:/RAGrepository 执行，使用同一虚拟环境。相关及全量测试均设置以下进程环境，不读取密钥配置：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:RUN_MODEL_TESTS = '0'
```

| 范围 | 实际命令 | 通过 | 失败 | 跳过 | 警告 |
|---|---|---:|---:|---:|---:|
| 新增 V2 回归 | .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_evaluation_dataset_v2.py | 27 | 0 | 0 | 0 |
| 相关数据与评估测试 | .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_retrieval_metrics.py tests/test_retrieval_evaluator.py tests/test_qa_evaluation.py tests/test_evaluation_dataset_v2.py | 149 | 0 | 0 | 1 |
| 全量收集、禁用真实模型测试 | .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short | 427 | 0 | 10 | 1 |

全量共收集 437 项。10 个跳过项均为 tests/test_embedder.py 的真实 BGE 模型集成测试，通过已有 RUN_MODEL_TESTS 开关跳过，没有删除或修改测试；其余历史测试继续通过。唯一 pytest 警告为既有 jieba 对 pkg_resources 的弃用提示。三个测试范围互有包含，不能将通过数相加。新增测试单独用时 0.12 秒，相关测试 2.45 秒，全量 75.60 秒。

本轮没有修改生产实现，遵循用户指定的先创建数据、再编写数据回归测试顺序；不将这一过程描述为生产代码的测试先行修复。

### 5.3 覆盖范围与复用

新增 tests/test_evaluation_dataset_v2.py 覆盖：13 个原始数据/历史文件的固定 SHA256；18/14 数量与现有加载器映射；恰好两个标签变化及 q02 保留；只允许的元数据更新；V1/V2 字节指纹；QA 语义内容不变；新旧配套成功与交叉配对失败；故意错误指纹仍拒绝；标签和空白字节变动即使不改正文也会破坏绑定；固定 ID 排名上的弱相关 Recall/RR/MRR/DCG/nDCG 手算兼容。

复用 tests/test_retrieval_metrics.py 的公式和边界测试，tests/test_retrieval_evaluator.py 的重复/未知 ID、JSON 字段、固定标签规则及报告防覆盖测试，tests/test_qa_evaluation.py 的必填字段、重复事实、非法文档引用、负例、SHA 错配及历史冻结保护测试。没有复制生产数据校验逻辑到新加载器，没有给生产代码增加测试专用接口。相关 evaluator 单元测试使用既有 fake 组件，不属于四种真实检索方案的完整实验。

### 5.4 最终保护与等效差异审查

测试后对执行前记录的 86 个文件逐项复核：0 个修改、0 个删除。两个原始数据集及 results 下全部 Day 7/8 历史文件共 13 个保护文件 SHA256 完全一致；此前独立审核报告也未改动。

只新增四个预定文件。工作区原有文件均未跟踪，默认 git diff 不能完整展示新增内容；因此采用全项目修改前后 SHA256 快照、JSON 结构差异及 git diff --no-index 对 V1/V2 成对核对。

检索文件文本差异仅四行现有元数据和 q04/q13 两行标签；QA 文本差异仅 version、corpus_sha256、metadata.name、metadata.corpus_file 四行。生产 app/、scripts/、原有 tests/、配置和依赖文件全部保持不变，默认评估输入路径不变。

Git 差异检查另提示未来可能将 LF 转为 CRLF；这不是本轮已发生的修改，实际字节哈希核对仍全部通过。由于当前采用字节指纹，未来格式化或换行转换后也必须重新核对配套绑定，不能忽略该提示或假定语义相同就哈希相同。

## 6. 修改范围与仍存限制

实际交付仅四个新增文件：retrieval_eval_v2.json、qa_eval_v2.json、本记录、tests/test_evaluation_dataset_v2.py；没有修改生产代码、已有测试或历史文件。

没有执行四方案评估、实际检索排名、真实 LLM 调用或模型下载。纯数据加载与对固定 ID 排名计算指标，不构成运行 Vector/BM25/Hybrid/Reranker 实验。

仍存限制：只有三项标签事项获用户裁决，其余人工审核未完成；18 条合成短文与少量问题仍不足以代表生产场景；参考答案与相关性标签不能由此次版本兼容检查自动认证。由于指纹绑定整个 JSON，未来即使只调整标签或排版，也需要维护 QA 的配套绑定。当前没有重构这一设计。
