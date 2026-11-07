# Docker 安装与镜像构建验收

日期：2026-11-06，Asia/Shanghai。

本轮按用户最新选择，仅安装 Docker 和构建镜像，不启动 RAG 应用，不配置真实 .env，不调用 LLM。

## 实际完成

| 项目 | 结果 |
|---|---|
| Docker Desktop | 4.98.0，用户级安装，目录 D:/DockerDesktop |
| WSL 后端 | WSL 2.7.3，Linux/amd64 Engine 正常响应 |
| Docker Engine | 29.9.2 |
| Docker Compose | v5.6.1 |
| Docker WSL 数据盘 | D:/DockerData/disk/docker_data.vhdx |
| 原 Ubuntu | 未迁移、未重新配置 |
| 镜像 | rag-knowledge-api:local |
| 镜像 ID | sha256:e57763d7ff903a5db5f307dce149c765fcf8f812197b4f1c6d94eac2c33634cc |
| 镜像大小 | 2,274,566,025 字节，约 2.12 GiB |
| 镜像运行用户 | 10001:10001 |
| 应用命令 | uvicorn app.deployment:create_deployment_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log |
| 构建内依赖校验 | pip check：No broken requirements found |
| 当前运行的 RAG 容器 | 0 |

安装包来自 WinGet 清单中的 Docker 官方下载地址，SHA256 为 cd6273d9eaa4f07fa7593fc6813d37e1004381cebc3f26047a2554fa5db24388，WinGet 哈希校验及 Docker Inc 数字签名均通过。安装采用官方提供的用户级、安装目录及 WSL 数据目录参数，参见 [Docker 官方安装说明](https://docs.docker.com/desktop/setup/install/windows-install/)。用户 PATH 已注册 Docker bin 目录；旧终端需要重新打开才能直接找到 docker。

## 构建过程

项目没有 .env，当前进程也未加载 API Key/Base URL。用户选择先构建，因此使用 Dockerfile 直接构建与 Compose 相同名称的镜像，没有创建假的运行配置或启动应用：

```powershell
cd D:\RAGrepository
docker build --progress=plain --tag rag-knowledge-api:local .
```

首次构建因旧终端 PATH 未包含新安装的凭据助手而失败；在构建进程 PATH 中补齐 D:/DockerDesktop/resources/bin 后解决。第二次构建在 Docker Hub 匿名认证端点连接超时。复用 Windows 已启用的系统代理到本次构建进程后成功，没有改系统代理或项目设置。

这台机器后续若遇到相同拉取超时，可在当前 PowerShell 中设置现有本地代理地址：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

这只是本机已有代理，并非为其他机器提供的固定地址。实际成功构建通过隔离子进程继承系统代理，未持久化代理环境变量。

基础镜像、系统库、CPU PyTorch 与 requirements 安装成功，pip check 通过，镜像导出与命名成功。构建上下文约 118.52 kB，没有复制评测数据、历史结果、模型权重或 .env。构建日志中的 root pip 警告仅发生在构建阶段，最终镜像 USER 已核实为 10001:10001。没有下载 BGE 或 Reranker 模型。

已删除本轮下载的临时安装包，释放约 595 MiB 系统盘空间；未清理其他文件或 Docker 数据。

## 范围与未执行项

- 未执行 Compose config/up；没有真实 .env，不把模板配置冒充已配置服务。
- 未启动 RAG 应用，因此本轮未验收 health、ready、Swagger、文档上传、容器重建恢复或真实问答。
- 未运行真实 LLM；未读取 .env、API Key 或 Authorization；未重新运行评测。
- 项目生产代码、Dockerfile 和依赖文件无修改，因此没有重复运行单元测试；本轮实测依据为 Engine 版本/信息、真实镜像构建、构建中的 pip check 和镜像 inspect。
- 新增验收记录及修改前哈希快照；共 291 个受保护代码/数据/结果文件核验，0 个变化，包括 V1/V2/V3 与 Day 7/Day 8 历史结果。

下一次实际应用部署需用户在本地准备现有 DeepSeek .env 和模型挂载路径，再运行 docker compose config --quiet 与 docker compose up -d。停止不用 down -v。当前系统盘和可用内存偏紧；未使用时可以退出 Docker Desktop 释放运行内存，镜像保留在 D 盘。
