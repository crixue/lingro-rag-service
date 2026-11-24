# test/

这里的文件是**集成脚本**，不是可独立运行的单元测试。

- 每个脚本都会连接真实的 MySQL / Redis / Elasticsearch / Neo4j / embedding 服务，并对线上同构的索引目录做读写；
- 路径统一放在 `./test_data/`（已被 `.gitignore` 忽略），运行前需要自行准备；
- 大多数脚本没有断言，输出是 `print` 结果，用于人工核对检索内容与相似度；
- 因此这些脚本**没有**纳入 CI。

## 运行方式

```bash
# 1. 准备 .env（参考仓库根目录 .env.example）
# 2. 确认外部服务可用
uv run pytest test/test_retriever_tools.py -s
```

直接用 `python` 运行也可以：`uv run python test/test_retriever_tools.py`。

## 脚本分类

| 类别 | 文件 |
|---|---|
| Agent / 工具 | `test_agent.py`、`test_langchain_agent.py`、`test_retriever_tools.py`、`test_file_system_tools.py`、`test_summary_tools.py`、`test_mcp_tools.py`、`test_tool_def.py` |
| 检索 | `test_retriever.py`、`test_bm25_document_index.py`、`test_elastic_search_client.py` |
| 索引构建 | `test_making_vectorstores_controller.py`、`test_making_hipporag_index_controller.py`、`test_documents_saver.py` |
| 文档加载与切分 | `test_document_loader.py`、`test_file_loader_v2.py` |
| 模型与 embedding | `test_llm.py`、`test_embeddings_v2.py` |
| 评测与 Prompt | `test_rag_checker.py`、`test_serialize_prompt.py`、`test_global_document_analysis.py` |

## 说明

测试夹具中的长文本是项目自写的合成内容，仅用于验证切分长度与语义相似度分布，不含任何第三方文章或业务数据。
