# 关键设计取舍

这份文档记录六个做过的技术选择：当时的问题、可选方案、最终选择和代价。面试追问大多会落在这里。

---

## 1. Agent 编排：从手写 ReAct 到 `create_agent`

**背景**：2024 年的实现是一个手写的 ReAct 循环（`core/agents/react/react_agent.py`），自己维护 scratchpad、解析 `Action: / Action Input:`、处理解析失败重试。代码仍保留在仓库里，用来对照演进。

**问题**：

- 每个模型的中文/英文、Gemini/GLM 的差异都要在 Prompt 里打补丁（`prompt_definitions/builtin/lang/**/react_system_prompt_for_*.json` 就是这段历史的痕迹）；
- 模型原生 tool calling 已经稳定，但手写循环还在走"文本里写 JSON"的老路，参数解析错误无法避免；
- 流式输出需要自己把中间状态变成 SSE，和最终答案的增量混在一起，前端难处理。

**选择**：迁移到 LangChain 1.x 的 `create_agent`（`service/chat_agent_service.py:224`），并把运行期需要的东西拆成三个概念：

| 概念 | 定义位置 | 内容 |
|---|---|---|
| Agent State | `context/state.py` | `retriever_config`、`context_tokens_limit`、`chat_session_state` |
| Runtime Context | `context/context.py` | `user_id`、`llm_client`，工具通过 `ToolRuntime` 读取 |
| 工具 | `tools_definitions/tools/` | 检索、文件读取、摘要、日期、MCP |

**收益**：

- 工具调用走模型原生 `tool_calls`，不再解析文本；
- `stream_mode="messages"` 直接给出 `AIMessageChunk` / `ToolMessage`，SSE 层只做格式转换；
- 工具无需闭包传参，`ToolRuntime` 注入 state 与 context，工具函数可独立测试；
- `recursion_limit` 统一由 `AGENT_MAX_ITERATIONS` 控制，配合 `GraphRecursionError` 兜底。

**代价**：依赖框架行为（中间件、状态合并语义），升级时需要回归；为此把版本精确锁定（`langchain==1.2.10`、`langgraph==1.0.9`）。

---

## 2. 工具返回拆分 content 与 artifact

**问题**：检索结果既要给模型看正文，又要给前端展示出处。如果都塞进 `content`，模型会看到大量 `doc_id / score / chunk_index` 之类的字段，既浪费 token，又容易在答案里复述元数据。

**选择**：所有检索类工具声明 `response_format="content_and_artifact"`（`retriever_tools.py:43`）。

```python
return formatted_docs, {"retriever_docs_info": list(docs_by_id.values()), ...}
```

- `content`：仅文档正文，按 token 预算拼接；
- `artifact`：`Document` 对象（含 metadata 与相似度），不进入模型上下文。

流式接口把引用作为独立的 `addition_info` 事件推给前端（`service/chat_message_util.py`）；非流式接口从 `ToolMessage.artifact` 里取出（`service/chat_agent_service.py:376`），并清洗掉对用户无意义的 metadata（`MetadataConstants.popup_useless_metadata`）。

**代价**：工具实现与调用方都要理解返回值结构；不同 LangChain 版本对 artifact 的传递方式有过变化，需要版本锁定。

---

## 3. Map-Reduce 摘要用两档模型

**问题**：一份 200 页的文档做摘要，逐段调用次数多（map），合并调用次数少（reduce）。如果全用最强模型，成本高；全用便宜模型，最终摘要质量差。

**选择**：map / reduce 使用不同产品 ID，且调用方可以覆盖（`tools_definitions/tools/summarize_tools.py`）：

| 阶段 | 默认产品 | 配置 | 理由 |
|---|---|---|---|
| map | `MODEL-0008`（qwen-flash） | `max_tokens=1800`、`temperature=0.1` | 次数多、任务单一 |
| reduce | `MODEL-0004`（qwen-max） | `max_tokens=4800`、`temperature=0.1` | 只调用一次，质量优先 |
| 文档分析 | `MODEL-0008` | `max_tokens=2400` | 结构化抽取，确定性要求高于文采 |

**代价**：两阶段摘要的中间结果需要落盘/缓存，避免重复消耗；文档变更后需要失效策略（按文件维度缓存）。

---

## 4. 单一模型工厂 + 供应商实现

**问题**：46 个模型产品横跨 8 类接入方式：部分走 OpenAI 兼容协议，部分是厂商自有协议（智谱 JWT、Gemini REST），推理模型还要额外处理 `reasoning_content`。

**选择**：`DefaultLangchainLLM.get_llm_client()` 按模型名分派（`core/model_io/llm_models/default_langchain_llm.py`）：

| 模型名特征 | 实现 | 备注 |
|---|---|---|
| `qwen*` | `tongyi_qwen_v4.ChatTongyi` | DashScope 兼容模式 |
| `glm-*` | `zhipu_glm_v3.ChatZhipuAI` | 自行实现 JWT 鉴权与流式解析 |
| `gemini-*` + `provider=google` | `google_gemini_v4.ChatGoogleGenerativeAI` | 走 REST |
| `gemini-*` + `provider=openrouter` | `openrouter_google_gemini_v4.ChatGoogleGemini` | 统一走 OpenRouter |
| `gpt-*` / `claude*` / `grok*` | `ChatOpenAI` 兼容实现 | 差异在 base_url 与 header |
| `deepseek*` | `deepseek_v4`（已标注废弃，工厂仍可达） | 后续由 `ChatOpenAI` 兼容实现替代 |

每个实现都挂载自己的 token 计量回调，因为 usage 字段结构不同：OpenAI 系有 `cached_tokens`，Anthropic 有 `cache_creation/cache_read`，智谱需要单独解析。

**代价**：分派靠字符串匹配，模型名不规范时会落到默认分支；长期应该改为显式 `provider` 字段 + 注册表（已在新模型配置中引入 `provider`，旧条目待迁移）。

---

## 5. 计费：前置校验 + 后置计量

**问题**：按 token 计费的服务，如果只在结束后扣费，会出现"余额不足但已经答完"的坏账；如果只做前置校验，长对话累计消耗可能超出余额。

**选择**：两段式。

1. **前置**：`pre_request_check`（`api_routers/llm.py:17`）调用 Java 的 `has-enough-money-for-service`，不足直接 402，不消耗模型调用；
2. **后置**：`BaseLLMTokenUsageCallbackFunction.on_llm_end`（`core/callbacks/llm_callbacks.py:107`）读取 `usage_metadata`，拆分 `input_tokens` / `output_tokens` / 缓存写入 / 缓存读取，然后 `PUT request-model-payment-txn`。

**取舍**：

- 上报失败只记日志、不阻断对话——计费链路的可用性不应影响主链路，缺口由对账补齐；
- 缓存 token 单独上报，因为供应商对缓存命中的计价不同，混在一起会导致对账差异；
- 前置校验是同步 `requests` 调用，在 `Depends` 里执行；这会占用事件循环线程，属于已知待优化点。

---

## 6. 混合检索的融合与后处理

**问题**：向量检索擅长语义相近，BM25 擅长专有名词与型号，图谱擅长多跳关系；单一后端都有盲区。

**选择**：

- 单路：`A01` 向量（+ BM25 双索引）、`A02` ES dense_vector + kNN、`A03` HippoRAG、`B01` 网络；
- 混合：`A01B01` / `A02B01` / `A03B01`，使用 LangChain `EnsembleRetriever(weights=fixed_weights)` 做加权融合（`mixin_retreiver_wrapper_v2.py:83`），权重由前端按场景传入，校验必须和为 1。

融合之后还有两个可选后处理：

| 参数 | 作用 | 适用场景 |
|---|---|---|
| `use_embedding_filter_assist` | `EmbeddingsFilter` + 冗余过滤，剔除低相关与重复片段 | 召回多但噪声大 |
| `use_reorder_assist` | `LongContextReorder`，把高相关文档放到上下文首尾 | 长上下文模型，缓解"中间信息丢失" |

最后统一按 `context_tokens_limit` 做预算截断：累计 token 超限即停止追加并记录日志（`retriever_tools.py:365`），而不是截断到半个文档。

**代价**：多路检索的延迟是叠加的（并发执行但最慢的一路决定耗时）；融合质量依赖权重调参，缺少自动寻优。

---

## 附：其他值得一提的小决定

| 决定 | 原因 |
|---|---|
| 索引构建同时支持同步与异步入口 | 被 HTTP 直接调用时要能在已有事件循环中执行，因此用线程池包一层（`making_index_entrance_controller.py:36`） |
| 索引版本号参与目录名与图库名 | 构建失败不影响线上版本，切换是原子的 |
| Prompt 与 JSON Schema 外置为文件 | 产品与运营可以改话术，不必发版；中英文各一套 |
| SSE 异常降级为文案 | 前端只需处理 `[DONE]`，不必区分流中断与正常结束 |
| 会话上传文件走独立索引 | 临时文件不应污染正式知识库，会话结束即删除 |
