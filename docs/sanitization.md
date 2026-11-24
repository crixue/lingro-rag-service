# 脱敏与改动说明

本仓库由生产仓库导出，**不携带任何 git 历史**（原历史中包含已废弃的密钥与内网地址）。本文列出导出后做过的全部处理，便于审阅与复核。

## 1. 移除的敏感信息

| 类别 | 内容 | 处理 |
|---|---|---|
| 模型 API Key | DeepSeek / OpenRouter / 通义千问 / 智谱 的访问密钥 | 从代码与配置中移除，改为环境变量；真实密钥已在生产侧轮换 |
| 数据库凭据 | MySQL 账号口令、Neo4j 口令 | 同上 |
| 服务凭据 | MCP 托管服务的 OAuth2 client secret | 同上 |
| 内网地址 | `192.168.x.x` 系列主机与端口、私有镜像仓库域名、内部代理地址 | 全部移除；代理改为可选环境变量 |
| 个人环境 | 开发者本机绝对路径、IDE 配置、旧版三环境配置文件 | 未进入副本；测试脚本中的路径改为 `./test_data/` |

复核方式（应全部为 0 命中）：

```bash
grep -rInE 'sk-[A-Za-z0-9]{16,}|192\.168\.|art\.local|bolt://|password\s*=\s*["'"'"'][^"'"'"']+' .
```

## 2. 移除的文件与目录

| 路径 | 原因 |
|---|---|
| `whls/`（3 个 wheel，85 MB） | 离线安装包，版本已过期且与 `pyproject.toml` 不一致 |
| `resources/get-pip.py` | 引导脚本，无业务价值 |
| `test/test_neo4j.ipynb` | 临时调试笔记本，含真实图数据库地址与口令 |
| `preprocess/` | 空包，仅剩 `__init__.py` |
| `core/model_io/langchain_google_genai/`（约 2500 行） | 早期 vendored 的第三方实现，全仓零引用；正式依赖 `langchain-google-genai>=2.1.7` 已安装 |
| `README-DEPLOY.md` | 内容过时且包含个人控制台链接，已重写为 [deployment.md](deployment.md) |
| `samples/`、`mock_data/`、`data/raw_data/`、`others/` | 真实文档、缓存、调试产物与离线二进制，未导出 |

## 3. 改写的内容

| 文件 | 改动 |
|---|---|
| `configs/local.env` → `.env.example` | 保留变量名与注释，值全部替换为占位符 |
| `Dockerfile` | 基础镜像由私有仓库地址改为 `ARG BASE_IMAGE`，附构建说明 |
| `base-nvidia-Dockerfile` | 改为基于公共 `nvidia/cuda` 镜像自建；不再依赖 `others/` 下的 deb / zip / wheel；NLTK 数据改为构建期下载；移除内网代理注释 |
| `deploy_on_dev.sh` / `deploy_on_prod.sh` | 仓库地址、应用名参数化（`REGISTRY` / `APP_NAME`），移除硬编码 |
| `core/agents/toolkits/wikipedia_tools/wikipedia.py` | 硬编码代理改为 `WIKIPEDIA_HTTP_PROXY` 环境变量 |
| `configs/model_config.py` | 内网默认地址改为 localhost 占位；`AAD_INFO.client_id` 改为环境变量；代理默认不启用 |
| `test/*.py`、`test/prompts/*.json` | 开发者本机路径统一为 `./test_data/`；三段长文夹具替换为合成的技术说明文本（保留长度与语义分布，用于切分与相似度测试） |
| `pyproject.toml` | 包名、描述、许可证元数据；`package = false` 使 `uv sync` 只解析依赖 |
| `.gitignore` | 追加 `.env*`、`logs/`、`test_data/`、`mock_data/` 等 |

## 4. 修复的缺陷

仅修复影响启动、正确性或安全性的小问题，业务逻辑未改动。

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | `api_routers/llm.py` `completions` | 同步函数内调用 `async` 方法，非流式接口必失败 | 改为 `async def` 并 `await` |
| 2 | `rag_checker/checker.py` | 从 `torch.fx` 导入 `logger`（错误来源，且引入无用依赖） | 改用项目 logger |
| 3 | `controllers/schedulers/build_knowledge_index_schedule.py` | Redis 不可用时后台任务持续抛异常 | 启动前 `ping` 探测，失败则降级退出；新增 `ENABLE_INDEX_SCHEDULER` 开关；读队列失败改为告警重试 |
| 4 | `utils/redis_client.py` | 未设置连接超时，Redis 不可达时请求长时间挂起 | 增加连接 / 读写超时 |
| 5 | `configs/model_config.py` | 缺任一环境变量即 `exit(1)`，本地无法启动 | `APP_ENV=local` 时降级为告警；生产仍强制校验 |
| 6 | `configs/model_config.py` | 日志目录写死 `/tmp/logs`，Windows 下落到系统盘根目录 | 支持 `LOG_FILE_DIR`，失败回退项目内 `logs/` |
| 7 | `configs/model_config.py` | 上游服务地址由 `os.getenv()` 直接字符串拼接，缺失时抛 `TypeError` | 统一用 `_service_base_url()` 提供占位默认值 |
| 8 | `app.py` | `allow_origins=["*"]` 与 `allow_credentials=True` 同时启用（浏览器会拒绝） | 改为 `CORS_ALLOW_ORIGINS` 配置，二者互斥 |
| 9 | `core/.../hipporag_with_graph_db.py` | `from sympy.codegen.cnodes import union` 无效导入 | 删除 |

## 5. 保留的已知问题

这些问题来自生产实现本身，公开版**没有**修复，以保持代码真实形态：

| 问题 | 位置 | 影响 |
|---|---|---|
| 索引类型分发不完整 | `controllers/making_index_entrance_controller.py:22` | `MIXED_*` 等类型会落到向量库分支 |
| MCP 无熔断、部分接口无超时 | `tools_definitions/mcp/mcp_tools.py` | MCP 不可用时 Agent 可能反复重试 |
| 队列消费用 `RPOP` | `controllers/schedulers/build_knowledge_index_schedule.py` | 进程异常导致在途任务丢失 |
| tool_call 累积器为进程内字典 | `service/chat_message_util.py:11` | 长连接需显式清理，多副本不共享 |
| 供应商分派依赖模型名字符串 | `core/model_io/llm_models/default_langchain_llm.py:75` | 模型命名不规范会落到默认分支 |
| 前置余额校验使用同步 `requests` | `api_routers/llm.py:26` | 占用事件循环线程 |
| 统一异常处理器擦除异常类型 | `exceptions/fastapi_exception_handlers.py` | 业务异常难区分，排障成本高 |
| 测试为集成脚本 | `test/` | 需真实中间件，未纳入 CI |

## 6. 未包含的内容

- 用户数据、企业文档、运营指标与日志；
- 前端、Java 微服务、Flutter 客户端等其它仓库；
- 数据库 DDL 与初始化数据；
- 真实评测数据集与线上评测结果。

## 7. 复核建议

1. 执行第 1 节的 grep，确认 0 命中；
2. `python -m compileall .` 全绿；
3. 检查 `git log` 只有一次初始提交，`git log -S 'sk-'` 为空；
4. 抽查 `.env.example` 中每个变量都在 `configs/model_config.py` 中有对应读取点。
