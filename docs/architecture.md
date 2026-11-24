# 架构说明

本文描述服务的分层、组件职责与四条主要调用链。所有路径均可在仓库中定位。

## 1. 系统上下文

```mermaid
flowchart TB
    subgraph Clients["调用方"]
        Web["Web / App / 浏览器插件"]
    end

    subgraph Service["lingro-rag-service · FastAPI"]
        direction LR
        API["HTTP API + SSE"] --> Agent["LangGraph Agent"] --> Ret["检索层"]
        Idx["索引流水线"]
        Eval["RAG 评测"]
    end

    subgraph Upstream["上游服务"]
        direction LR
        Java["Java 微服务<br/>计费 / 额度 / 版本 / 网页检索"]
        LLM["LLM 供应商<br/>Qwen · DeepSeek · GLM · OpenRouter"]
        Embedding["Embedding 服务<br/>OpenAI 兼容"]
        MCP["MCP Server(s)<br/>stdio / sse"]
    end

    subgraph Infra["基础设施"]
        direction LR
        MySQL[("MySQL<br/>会话 / 配置")]
        Redis[("Redis<br/>任务队列")]
        ES[("Elasticsearch<br/>全文 + 向量")]
        Neo4j[("Neo4j<br/>知识图谱")]
        FAISS[("FAISS + BM25<br/>本地索引")]
    end

    Web -->|"userid / productId"| API
    Agent -->|"推理"| LLM
    Agent -->|"工具调用"| MCP
    Agent -->|"token 计量"| Java
    API -->|"余额预检"| Java
    API --> Eval
    Ret -->|"检索"| ES
    Ret --> Neo4j
    Ret --> FAISS
    Ret -->|"向量化"| Embedding
    Idx -->|"写入"| ES
    Idx --> Neo4j
    Idx --> FAISS
    Idx -->|"任务队列"| Redis
    API --> MySQL

    classDef svc fill:#e8f1ff,stroke:#3b82f6,stroke-width:2px,color:#123
    classDef store fill:#fff7ed,stroke:#f59e0b,stroke-width:1.5px,color:#123
    classDef ext fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#123
    class API,Agent,Ret,Idx,Eval svc
    class MySQL,Redis,ES,Neo4j,FAISS store
    class Java,LLM,Embedding,MCP,Web ext
```

图片版：[diagrams/system-context.png](diagrams/system-context.png)

**边界**：本服务不负责用户注册、登录、权限与支付账本，只消费网关透传的 `userid` 与 `productId`，并把余额校验、令牌计量交给上游 Java 服务。

## 2. 分层结构

```mermaid
flowchart TB
    subgraph L1["api_routers"]
        A1["llm"]
        A2["local_knowledge_base_actions"]
        A3["mcp"]
        A4["rag_checker"]
        A5["prompt"]
        A6["plugins"]
        A7["fastapi_middleware / context"]
    end

    subgraph L2["controllers + service"]
        B1["BuildingIndexEntranceController"]
        B2["dispatch_handlers<br/>向量 / ES / 图谱"]
        B3["schedulers"]
        B4["DefaultChatAgentService"]
        B5["chat_message_util"]
    end

    subgraph L3["core"]
        C1["model_io<br/>工厂 / 供应商 / Prompt"]
        C2["data_connections<br/>加载 / 切分 / 检索"]
        C3["callbacks<br/>计量"]
        C4["agents<br/>早期 ReAct"]
    end

    subgraph L4["utils / context / exceptions"]
        D1["MySQL / Redis / ES / Neo4j"]
        D2["Agent State / Runtime Context"]
        D3["异常与统一响应"]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
```

## 3. 对话调用链

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant MW as UserContextMiddleware
    participant R as llm router
    participant J as Java 微服务
    participant F as DefaultLangchainLLM
    participant S as DefaultChatAgentService
    participant G as LangGraph Agent
    participant T as 工具 / 检索器
    participant P as LLM 供应商

    C->>MW: POST /chat-stream/completions (userid, productId)
    MW->>MW: 写入 userid 到 context
    MW->>R: 转发请求
    R->>J: GET has-enough-money-for-service
    J-->>R: true / false
    alt 余额不足
        R-->>C: 402 Not have enough money
    else 余额充足
        R->>F: 按 model_id 构造模型客户端
        F-->>R: BaseChatModel + 计费回调
        R->>S: completions_stream(chat_request)
        S->>S: 按语言 / 上下文 / 插件组装 system prompt
        S->>S: 组装工具（检索 / 文件 / 总结 / MCP）
        S->>G: create_agent(...).astream(...)
        loop 直到模型给出最终答案或达到 recursion_limit
            G->>P: 推理
            P-->>G: 文本增量 / tool_call
            G->>T: 执行工具
            T-->>G: ToolMessage + artifact
        end
        G-->>S: 消息流
        S-->>C: SSE content / tool_call / tool_message
        S-->>C: addition_info（引用文档）
        S-->>C: data: [DONE]
        F->>J: 上报 token 用量（on_llm_end 回调）
    end
```

图片版：[diagrams/chat-sequence.png](diagrams/chat-sequence.png)

关键实现：`api_routers/llm.py`、`api_routers/fastapi_middleware.py`、`service/chat_agent_service.py`、`core/callbacks/llm_callbacks.py`。

## 4. 索引构建

```mermaid
flowchart TB
    Q["POST /knowledge-index/v1/build"] --> E{"BuildKnowledgeIndexTypeEnum"}
    E -->|A01| V["BuildingVectorStoreIndexStrategy"]
    E -->|A02| S["BuildingElasticSearchIndexStrategy"]
    E -->|A03| H["BuildingHippoRAGGraphIndexStrategy"]

    V --> L
    S --> L
    H --> L

    L["文件加载<br/>unstructured / OCR / 语音转写文本"]
    L --> SP["文本切分<br/>中英文 chunk / overlap / 父子块"]
    SP --> EM["向量化<br/>Embedding 服务，分批 + 重试"]
    EM --> W1["写入 FAISS 索引 + BM25"]
    EM --> W2["写入 Elasticsearch"]
    EM --> W3["OpenIE + 实体消歧 + 写入 Neo4j"]
    W1 --> DS["保存 docstore"]
    W2 --> DS
    W3 --> DS
    DS --> N["回调 Java 服务<br/>notice-version-checkout"]
    N --> SW["版本切换"]
```

定时 / 异步入口：`controllers/schedulers/build_knowledge_index_schedule.py`（Redis 队列，两个消费者分别处理整包构建与单文件构建）。类型分发的已知缺口见 [sanitization.md](sanitization.md)。

## 5. 检索路由

```mermaid
flowchart LR
    IN["RetrieverConfig.retriever_type"] --> F{"RetrieverWrapperFactory"}

    F -->|A01| V["LocalVectorStoreRetrieverWrapperV2"]
    F -->|A02| S["ElasticSearchRetrieverWrapperV2"]
    F -->|A03| H["HippoRAGRetrieverWrapperV2"]
    F -->|B01| W["WebSearchVectorStoreRetrieverWrapperV2"]
    F -->|A01B01| MV["MixedLocalVSAndWebSearch"]
    F -->|A02B01| MS["MixedElasticsearchAndWebSearch"]
    F -->|A03B01| MH["MixedHippoKGAndWebSearch"]

    V --> F1["FAISS 相似度检索<br/>父块召回 + 子块返回"]
    V --> F2["BM25 稀疏检索"]
    S --> F3["dense_vector kNN + 过滤"]
    H --> F4["实体抽取 → PPR 传播 → 文档排序"]
    W --> F5["Java 网页检索服务<br/>多 embedding 加权采样"]

    F1 --> M["加权融合 / 去重 / 可选重排"]
    F2 --> M
    F3 --> M
    F4 --> M
    F5 --> M
    M --> D["按 context_tokens_limit 截断"]
    D --> R["返回 content + artifact"]
```

实现：`core/data_connections/retrievers/default_retriever_wrapper_v2.py`（工厂）、`mixin_retreiver_wrapper_v2.py`（混合检索）、`knowlege_graph_retrievers/hipproRAG/`（图谱）。

## 6. 模型接入与计费

```mermaid
flowchart TB
    CFG["SUPPORTED_MODELS<br/>46 个产品配置"] --> U["SupportedModelsInfoUtil.find_supported_model_by_id"]
    U --> FACT["DefaultLangchainLLM.get_llm_client"]

    FACT -->|name 含 qwen| Q["ChatTongyi"]
    FACT -->|name 含 glm-| Z["ChatZhipuAI"]
    FACT -->|gemini + google| GE["ChatGoogleGenerativeAI"]
    FACT -->|gemini + openrouter| GO["ChatGoogleGemini"]
    FACT -->|gpt- / claude / grok| O["ChatOpenAI 兼容实现"]
    FACT -->|deepseek| DS["ChatDeepSeek"]

    Q --> CB["token 计量回调"]
    Z --> CB
    GE --> CB
    GO --> CB
    O --> CB
    DS --> CB

    CB --> AGG["聚合 usage_metadata<br/>input / output / cache read / cache write"]
    AGG --> PUT["PUT request-model-payment-txn"]
    PUT --> BILL["Java 计费服务"]

    PRE["pre_request_check"] --> PAY["GET has-enough-money-for-service"]
    PAY -->|不足| E402["402"]
```

## 组件职责与位置

| 组件 | 职责 | 位置 |
|---|---|---|
| 路由层 | 参数校验、鉴权头提取、SSE 响应封装 | `api_routers/` |
| 中间件 | 把 `userid` 写入请求上下文，供工具与回调读取 | `api_routers/fastapi_middleware.py`、`api_routers/context.py` |
| Agent 服务 | 工具组装、Prompt 组装、流式/非流式调用、兜底语气 | `service/chat_agent_service.py` |
| SSE 格式化 | 文本增量、tool_call 参数拼接、工具消息、引用元数据 | `service/chat_message_util.py` |
| 索引入口 | 按类型选择构建策略，线程/事件循环兼容 | `controllers/making_index_entrance_controller.py` |
| 构建策略 | 加载 → 切分 → 向量化 → 写入后端 → 保存 docstore | `controllers/dispatch_handlers/` |
| 检索器 | 4 类后端 + 3 种混合组合 | `core/data_connections/retrievers/` |
| 模型工厂 | 按模型名分派供应商实现 | `core/model_io/llm_models/default_langchain_llm.py` |
| 工具 | 检索、文件读取、摘要、日期、知识库信息 | `tools_definitions/tools/` |
| MCP 桥接 | 配置加载、连接、工具转换 | `tools_definitions/mcp/`、`service_communication/` |
| 评测 | 12 项指标计算与 LLM Judge | `rag_checker/` |
| 状态定义 | Agent State 与 Runtime Context | `context/` |

## 数据存放

| 数据 | 位置 | 说明 |
|---|---|---|
| 会话 / 配置 | MySQL | 无 ORM，`PyMySQL` 直连 |
| 索引任务 | Redis List | 两个队列：整包构建、单文件构建 |
| 全文与向量 | Elasticsearch | `dense_vector` 维度需 ≤ 4096 |
| 知识图谱 | Neo4j | 每版本一个图库，前缀区分 |
| 本地向量 | FAISS 文件 | 按 `workspace_id` 分目录 |
| 文档存储 | docstore 文件 | 与 FAISS 索引同目录，用于父块回填 |
