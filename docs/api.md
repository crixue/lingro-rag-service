# API 参考

服务没有统一的 context-path，各路由在 `app.py` 中直接注册。基准地址示例：`http://localhost:8000`。

所有非健康检查接口都依赖网关传入的请求头：

| 请求头 | 说明 |
|---|---|
| `userid` | 用户标识，缺失时对话接口返回 401 |
| `productId` | 模型产品 ID（如 `MODEL-0002`），用于余额校验与计费 |

---

## 1. 对话

### 1.1 流式对话

`POST /llm-backend/v1/chat-stream/completions`

**请求体**（`ChatRequestVO`）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 本轮用户输入 |
| `history_messages` | array | 是 | 至少一条；建议首条为 `system`，Agent 会向其追加 ReAct 指令 |
| `used_model_config` | object | 是 | `llm_id`、`is_chinese_text`、`have_context`、`model_config` |
| `retriever_config` | object | 是 | 检索类型、知识库目录、版本、top_k 等，见下表 |
| `used_mcp_tools` | array | 否 | 本次请求启用的 MCP 服务器与工具 |
| `chat_session_upload_file_info` | object | 否 | 会话临时上传文件；存在时改用"上传文件检索 / 总结"工具集 |
| `used_functions` | array | 否 | 仅非流式生效 |

**`retriever_config` 主要字段**

| 字段 | 默认 | 说明 |
|---|---|---|
| `retriever_type` | `A01` | `A01` 向量 / `A02` ES / `A03` 图谱 / `B01` 网络 / `A01B01` / `A02B01` / `A03B01` |
| `knowledge_index_folder_id` | — | 知识库目录 ID；`B01` 可不传 |
| `knowledge_index_folder_name` | `web_search` | 本地索引目录名 |
| `version` | `-1` | 索引版本，`-1` 表示最新 |
| `search_type` | `similarity` | `similarity` 或 `mmr` |
| `search_top_k` | `5` | 召回条数 |
| `web_search_results_count` | `20` | 网络检索条数 |
| `fixed_weights` | `[0.5, 0.5]` | 混合检索权重，必须和为 1 |
| `use_embedding_filter_assist` | `false` | 追加相似度过滤 |
| `use_reorder_assist` | `false` | 追加 LongContextReorder |
| `ignore_content_len` | `120` | 过短片段过滤阈值 |

**响应**：`text/event-stream`，每个事件为 `data: <json>\n\n`，最后一条为 `data: [DONE]`。

事件体与 OpenAI `chat.completion.chunk` 兼容，额外字段如下：

| `type` | 附加字段 | 含义 |
|---|---|---|
| `content` | `content` | 答案文本增量 |
| `reasoning_content` | `content` | 思维链增量（仅推理模型） |
| `tool_call` | `tool_call: {name, args, id, index, status}` | 工具调用参数已拼接完整 |
| — | `tool_message` | 工具执行结果（含状态与截断标记） |
| `addition_info` | `retriever_docs`、`search_keywords` | 引用文档与检索关键词，供前端渲染出处 |

```json
{
  "id": "msg_xxx",
  "object": "chat.completion.chunk",
  "model": "qwen3.6-plus",
  "type": "addition_info",
  "retriever_docs": [
    {"page_content": "...", "metadata": {"source": "handbook.pdf", "page": 12, "score": 0.83}}
  ],
  "search_keywords": "支持的文件格式"
}
```

### 1.2 非流式对话

`POST /llm-backend/v1/chat/completions`

请求体同上。响应为 `chat.completion`，`choices[0].message.content` 为答案，另含：

- `retriever_docs`：引用文档（从检索工具的 `artifact` 中提取）
- `used_function_ids`：本轮调用过的工具名

### 1.3 Mock 流式对话

`POST /llm-backend/v1/mock-chat-stream/completions`

从 `mock_data/` 随机读取一个 SSE 录制文件并按行回放，用于前端联调，不调用模型。该目录不在公开快照中，使用前需自行放置样例数据。

---

## 2. 知识库索引

### 2.1 构建

`POST /knowledge-index/v1/build`

| 字段 | 别名 | 说明 |
|---|---|---|
| `id` | — | 任务 ID |
| `build_knowledge_index_type` | `buildKnowledgeIndexType` | `A01` / `A02` / `A03` |
| `raw_file_type` | `rawFileType` | `DEFAULT` / `CONTENT_STRING` / `URL` / `SPEECH_RECOGNIZE_TRANSCRIPT` / `OCR_PARSE_TRANSCIPT` |
| `save_to_path` | `saveToPath` | 源文件路径或内容 |
| `knowledge_index_folder_id` | `knowledgeIndexFolderId` | 工作区 ID（必填） |
| `knowledge_index_folder_name` | `knowledgeIndexFolderName` | 索引目录名 |
| `user_id` | `triggerUserId` | 触发用户 |
| `is_chinese_text` | `isChineseText` | 决定切分参数与 Prompt 语言 |
| `checkout_version` | `checkoutVersion` | 目标版本号 |
| `build_knowledge_index_config` | `buildKnowledgeIndexConfig` | 切分 / 距离策略等 |
| `token_limits` | `tokenLimits` | 构建期上下文上限 |
| `enable_analyze_document` | `enableAnalyzeDocument` | 是否生成文档级分析与摘要 |

### 2.2 删除

| 方法 | 路径 | 说明 |
|---|---|---|
| DELETE | `/knowledge-index/v1/drop` | 删除指定索引 |
| DELETE | `/knowledge-index/v1/drop-version-of-kg-index` | 删除某版本的图谱索引与文档 |
| DELETE | `/knowledge-index/v1/delete-workspace` | 删除整个工作区 |
| DELETE | `/knowledge-index/v1/chat-session/drop?chat_session_id=&user_id=` | 删除会话临时索引 |

---

## 3. MCP

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/mcp-servers/v1/tools/list` | 列出指定 MCP Server 的工具 |
| POST | `/mcp-servers/v1/prompts/list` | 列出 Prompt |
| POST | `/mcp-servers/v1/prompt/get` | 获取指定 Prompt |
| POST | `/mcp-servers/v1/resources/list` | 列出 Resource |
| POST | `/mcp-servers/v1/resources/templates/list` | 列出 Resource 模板 |

请求体为 `MCPServer`（`controllers/vo/mcp_connection.py`），支持 `stdio` 与 `sse` 两种传输方式，托管服务使用 OAuth2 client credentials 获取 access token。

---

## 4. 评测与 Prompt

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/rag-checker/v1/evaluate` | 触发一次 RAG 评测，请求体见 `controllers/vo/rag_checker_vo.py` |
| PUT | `/prompt/v1/serialize` | 将 Prompt 模板序列化后写入指定目录 |

---

## 5. 健康检查

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health/liveness` | 存活探针：服务就绪标志 + MySQL + Redis |
| GET | `/health/readiness` | 就绪探针，判定逻辑同上 |

返回示例：

```json
{
  "status": "alive",
  "checks": {
    "service": {"status": "ok"},
    "mysql": {"status": "ok"},
    "redis": {"status": "error", "detail": "Connection refused"}
  }
}
```

任一检查失败时返回 HTTP 503。

---

## 错误语义

| 状态码 | 场景 |
|---|---|
| 401 | 缺少 `userid` 请求头 |
| 402 | 余额不足，请求未进入模型调用 |
| 422 | 请求体校验失败（Pydantic） |
| 500 | 未捕获异常；具体异常类型会被统一异常处理器包装 |

流式接口在模型层异常时不返回错误码，而是在 SSE 中推送一条兜底文案（区分中英文与限流场景），随后正常以 `[DONE]` 结束——这样前端无需为流中断单独处理。
