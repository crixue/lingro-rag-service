# 部署说明

## 1. 运行形态

服务是一个无状态 FastAPI 应用，所有状态在 MySQL / Redis / Elasticsearch / Neo4j / 卷上。

| 项 | 值 |
|---|---|
| 进程 | `uvicorn app:app --host 0.0.0.0 --port 80` |
| 健康检查 | `GET /health/liveness`、`GET /health/readiness`（任一依赖不可用返回 503） |
| 监听端口 | 容器内 80，宿主机按需映射 |
| 时区 | 默认 `Asia/Shanghai`，可用 `SERVER_TIMEZONE` 覆盖 |

## 2. 卷挂载

镜像声明了 8 个卷，部署时需要持久化映射（名称需与容器内路径一致）：

| 容器内路径 | 用途 | 必需 |
|---|---|---|
| `/vector_store` | FAISS 索引与 docstore | 是（A01 检索） |
| `/bm25_index` | BM25 倒排文件 | 是（A01 + BM25） |
| `/prompt_templates` | 运行时可热更新的 Prompt 模板 | 否 |
| `/logs` | 应用日志 | 否（可用 `LOG_FILE_DIR` 改） |
| `/data` | 原始文档缓存 | 否 |
| `/models` | 本地模型文件（如需要） | 否 |
| `/tmp` | 临时文件、会话上传文件 | 是 |
| `/make_kb_retryable_context_file_dir` | 索引失败重试上下文 | 否 |

> 多副本部署时，`/vector_store` 与 `/bm25_index` 不能是本地盘，否则同一工作区的索引会分散在不同 Pod 上。生产做法是为每个工作区固定路由到单一副本，或改用共享文件系统。

## 3. 系统依赖

基础镜像里已经安装，自建镜像时不要漏：

```bash
apt-get install -y \
  libmagic-dev poppler-utils libreoffice pandoc \
  tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
  pkg-config cmake build-essential
```

- `libmagic` / `poppler`：文件类型识别与 PDF 解析
- `libreoffice`：Office 文档转换（pptx/docx/xlsx）
- `pandoc`：部分格式的二次转换
- `tesseract` + 中文语言包：扫描件 OCR

`unstructured[all-docs]` 在首次处理某些格式时会下载模型，建议在构建镜像阶段预热，避免运行时阻塞。

## 4. 构建镜像

依赖镜像（可复用，包含 CUDA、Python 3.12、系统库、Python 依赖）：

```bash
docker build -t registry.example.com/lingro-rag-deps:latest -f base-nvidia-Dockerfile .
```

应用镜像（只拷贝代码，构建快）：

```bash
docker build \
  --build-arg BASE_IMAGE=registry.example.com/lingro-rag-deps:latest \
  -t registry.example.com/lingro-rag-service:latest -f Dockerfile .
```

无 GPU 环境：

```bash
docker build \
  --build-arg CUDA_IMAGE=ubuntu:24.04 \
  --build-arg FAISS_SPEC=faiss-cpu \
  -t registry.example.com/lingro-rag-deps:latest -f base-nvidia-Dockerfile .
```

## 5. 配置注入

不要把 `.env` 打进镜像。推荐在运行时注入：

```bash
docker run -d --name rag-service -p 8080:80 \
  --env-file /etc/lingro/rag.env \
  -v /srv/rag/vector_store:/vector_store \
  -v /srv/rag/bm25:/bm25_index \
  registry.example.com/lingro-rag-service:latest
```

Kubernetes 中建议用 `Secret` + `envFrom`，并把连接串写成 `ConfigMap`。必需变量清单见 `.env.example` 顶部。

## 6. 发布脚本

`deploy_on_dev.sh` / `deploy_on_prod.sh` 是内部发布流程的简化版，已参数化：

```bash
# 构建并推送 latest（可选打版本标签）
REGISTRY=registry.example.com/docker-local TAG_VERSION=v1.2.0 ./deploy_on_dev.sh

# 把 latest 提升为发布版本并打 git tag
REGISTRY=registry.example.com/docker-release INITIAL_VERSION=v1.2.0 ./deploy_on_prod.sh
```

两个脚本都会清理本地旧版本镜像，`deploy_on_prod.sh` 额外创建并推送 git tag。真实生产还会涉及蓝绿切换与回滚，这部分属于编排层，未包含在本仓库。

## 7. 首次启动检查清单

1. `GET /health/liveness` 返回 200；
2. 日志中确认 `Service is ready.`；
3. 若使用 Redis 队列，确认 `ENABLE_INDEX_SCHEDULER=true` 且队列名与上游一致；
4. 触发一次小文件索引构建，确认 ES 索引 `dense_vector` 维度 ≤ 4096；
5. 调用一次流式对话，确认 SSE 能收到 `[DONE]`；
6. 检查 Java 侧是否收到 token 计量请求（计费链路是否连通）。

## 8. 已知运维事项

| 事项 | 说明 |
|---|---|
| 索引任务丢失 | 队列用 `RPOP` 消费，进程崩溃会丢在途任务；需配合重试接口与人工补偿 |
| 图谱构建耗时 | OpenIE 抽取按文档串行批处理，大文档集建议错峰调度 |
| 首次检索延迟 | FAISS 索引按需加载，冷启动首次查询较慢 |
| 日志增长 | 默认按天切割保留 7 天，可用 `LOG_FILE_DIR` 指向持久卷 |
