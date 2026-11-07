# Docker 真实运行验收

日期：2026-11-07，Asia/Shanghai。

## 实际结果

| 检查 | 结果与证据 |
|---|---|
| Compose 配置 | `docker compose config --quiet` 返回 0 |
| 启动 | `up -d --no-build --pull never` 成功，使用原有镜像 |
| /health | HTTP 200，status=ok |
| /ready | HTTP 200，status=ready；部署入口已经加载真实本地 Embedding 和 Cross-Encoder |
| /docs 与 OpenAPI | HTTP 200，Swagger UI 标记存在，健康/就绪/上传/问答路由存在 |
| 文本型 PDF 上传 | HTTP 201，indexed，945 字节、1 个 Chunk |
| 第一次真实问答 | HTTP 200，answered，4.172 秒；两项测试值均出现，实际来源包含上传 PDF |
| 强制重建 | 容器 ID 改变，镜像 ID 未改变；没有构建或拉取镜像 |
| 重建后再次问答 | HTTP 200，answered，3.375 秒；两项测试值和上传来源仍然存在 |
| 磁盘持久化 | 上传 PDF 与 manifest.json 重建前后 SHA256 完全一致 |
| 重建后重复上传 | HTTP 200，duplicate，同一 document_id，磁盘文件未变化 |
| 最终容器状态 | running / healthy，127.0.0.1:8000→8000 |

本次使用用户在本地填写的现有 LLM 配置，是真实模型调用，未使用 MockLLM。共 2 次逻辑问答请求，全部成功；没有单独计量 SDK 底层 HTTP 尝试次数或 token 用量。没有在报告或应用日志中保存完整模型回答。

## 验收 PDF 与来源

使用独立的合成文本 PDF `docker-acceptance-20261107.pdf`，没有改动评测集。PDF 明确标记这些是验收标签，而非生产配置：

- Orion launch code：RAG-PDF-7821。
- verification window：47 minutes。

上传 document_id：`bfa523d6243c45fc8eca7e7ea82dae4f`。两次回答的结构化来源均含该 ID、原文件名、chunk_id=0、page=1；第一次排序为 rank=1。来源正文包含这两项已知测试事实。

两次使用同一个问题，要求根据该 PDF 简短给出校验码和时间窗口。验收检查了非空 answered、两个预期字段、无明显缺证拒答短语及对应来源。结论只覆盖当前部署样本，不代表所有问题的答案质量已经合格。

PDF SHA256：`5af21a0acf17a2aa0698b3f4e1f83641f7ea90a30abf8ca492a4b67ed628753e`。

manifest SHA256：`296a9f1f2495106701fd6755f520f96abe1f01dd70517ce09fdc77f859288b61`。

## 容器重建与挂载

实际执行：

```powershell
docker compose up -d --force-recreate --no-build --pull never rag-api
```

旧容器 ID：`153c20cd73cacbd734f6a7fa674fc7fda56df963cc92ef6c34bd03633668dfa8`。

新容器 ID：`ae40b3a46bbca9e3b7868e82d7f329c9091cf4a8f38cf98e8bd4f1f56c4dc4f5`。

镜像保持：`sha256:e57763d7ff903a5db5f307dce149c765fcf8f812197b4f1c6d94eac2c33634cc`。

| 挂载 | 实际宿主机路径 | 权限 |
|---|---|---|
| /app/data/uploads | D:/RAGrepository/data/uploads | 读写 |
| /app/knowledge | D:/RAGrepository/data | 只读 |
| /models/hub | C:/Users/吴志武/.cache/huggingface/hub | 只读 |

恢复后先执行问答，再进行重复上传检查，因此不是靠重新上传掩盖恢复失败。验收文件及 manifest 保留在上传目录，可继续演示。没有执行 down -v。

## 配置与保护

原先不存在 .env，本轮先创建了无密钥模板并填入公开的模型/目录信息，再由用户在本机填入现有 Provider 配置。配置通过 Compose 注入，没有显示、复制或保存密钥到验收记录；没有手动读取 .env 内容或输出容器完整环境。

未重新构建镜像、未修改算法或生产代码、未下载模型、未运行四方案评估。291 个受保护代码/数据/历史结果文件的 SHA256 全部不变。未重复运行原有单元测试，本轮验证针对真实容器和真实问答。

## 限制与自动安全检查

自动安全检查拒绝了包含完整日志检查、容器内身份查询和自动打开浏览器的合并命令，返回 `blocked by policy`，未说明更具体的原因。没有重试该合并操作；后续只查询了只读容器状态并核验已有记录。完整日志审查和自动浏览器打开未完成，Swagger 通过 HTTP 响应与页面标记验证。

人工查看地址：http://localhost:8000/docs 。本轮五项功能验收已完成，容器保持运行，不进入下一阶段。

机器可读总记录：`docker_runtime_acceptance_2026-11-07.json`。逐步证据保存在同目录的 docker_runtime_preflight/upload/chat_before/chat_after/duplicate/container_before 等 JSON 文件中。
