# V3 第一次 AI 辅助评审协议（2026-10-27）

这是用户授权 Codex 代评的离线语义评审，不是用户或独立人工评分；不调用真实 API，不产生新模型答案，不修改采集数据和标签。

唯一评分依据：docs/day8_qa_evaluation.md 与 app/evaluation/qa_review.py 的既有规则。faithfulness、answer_relevance、factual_correctness 各自 0/1/2，不合成总分。先逐份读实际 answer 和 context_documents，再对照 reference_answer、required_facts、冻结 18 Chunk；不得用其他方案的正文冒充该答案的实际 context。

- faithfulness：2 = 事实性断言由实际上下文支持，允许合理同义转述和直接逻辑推论；1 = 局部无依据扩展或限制不充分；0 = 明显核心无依据断言或与上下文矛盾。无依据断言必须逐条摘原文并指出实际 context 缺失何种支撑。不能仅因该事实存在于知识库其他未提供的 Chunk 就给忠实度 2。
- answer_relevance：2 = 直接回应核心问题（证据不足时解释不能确定以及具体缺失信息）；1 = 只回答问题的一部分、过度拒答或明显偏题；0 = 没有回应。符合上下文的部分拒答可保持忠实度 2，但对于知识库可回答而此次上下文缺失的题，仍应反映任务未完成，通常切题性 1，而非把拒答当成正确完整回答。
- factual_correctness：2 = 核心事实正确，没有实质事实错误；1 = 局部有误，主要事实正确；0 = 关键结论错误或严重误导。缺失事实不直接等同事实错误；依据必须写明冻结参考答案、具体语料 ID 和限制。CUDA 编号只按示例日志范围核对，不当作官方错误码定义。不新增未经核查的外部权威认证。
- required_facts_covered：逐项语义判断，不按关键词机械评分。true 必须有真实答案原文摘录；部分复合事实缺关键部分为 false 并解释，语义等价可覆盖，不要求复制参考答案措辞。仅“没有某个具体值”不自动覆盖“需要当前实际检测/识别核验/计时对照/编码输出”的行动或证据获取要求。模型明确指出缺少具体测量/输出记录时，可作为相应缺证事实的覆盖，但不能把普通数值缺失强行解释成已要求取得实测结果。
- unsupported_claims：无则 []；有则 claim/answer_quote/reason/context_document_ids，ID 只能引用实际 context。现实中正确但此次 context 无依据仍须记录；不要惩罚只是自然语言解释性推论。
- insufficient_evidence_handled：对应用户的 Evidence Sufficiency，answerable=false 时必填 bool，answerable=true 保留 null（未增设新分数）。判断是否说明资料不足、避免无依据确定结论，并适当说明具体缺失信息。说明实际缺失信息可以是缺口解释，不强制使用“请提供”字样。是否覆盖补充取证行动另记 required_facts_covered。不要把非空 context、pipeline_status=answered 或分数视为证据充分。

所有 completed 仅表示这份 AI 评审字段已完成；reviewer 必须显式为 Codex AI，不署用户名或“人工”。评分只写新副本，不修改 results/v3_qa_review/*、v3_qa_comparison、进度、Day 7/8 或数据集。新增摘要不得把 AI 评分放在标作 human_review_metrics 的结果中。3 个调用失败不评分、不补答案。

第一轮 32 个负例；第二轮 165 个成功的可回答样本。统一尺度，不根据方法成绩调整。需要标记有争议的语义边界，供用户复核。事实正确性范围是教学语料与参考答案的本地核对，不是额外查证厂商文档。
