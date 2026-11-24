import unittest

from langchain.agents import create_agent

from configs.model_config import AGENT_MAX_ITERATIONS
from context.context import ApplicationContext
from context.state import ChatState
from controllers.vo.startup_llm_model_request_vo import RetrieverConfig, ChatSessionUploadFileInfo, ChatSessionState
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from tools_definitions.tools.retriever_tools import search_relevant_documents_from_knowledge_base_tool, \
    search_relevant_documents_from_uploaded_files_tool


class TestRetrieverTools(unittest.TestCase):

    def setUp(self):
        """测试前的初始化设置"""
        self.user_id = "92040b49"

        # 配置模型参数
        model_config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "streaming": True,
            "max_tokens": 2000,
        }

        # 创建 LLM 客户端
        self.chat_model = DefaultLangchainLLM.llm_client_factory(
            product_id="MODEL-0008",
            llm_model_name="qwen-flash",
            user_id=self.user_id,
            llm_model_config=model_config,
            additional_llm_config={"parallel_tool_calls": False}
        )

        # 创建应用上下文
        self.application_context = ApplicationContext(
            user_id=self.user_id,
            llm_client=self.chat_model
        )

        # 配置检索器参数
        self.retriever_config = RetrieverConfig(
            is_chinese_text=True,
            retriever_type="A01",
            knowledge_index_folder_id='af60402d',
            knowledge_index_folder_name='af60402d/1761379771990',
            version='1761379771990'
        )

        # 上下文 token 限制
        self.context_tokens_limit = 4000

    def test_search_relevant_documents_with_agent(self):
        """测试使用 Agent 调用检索工具搜索相关文档"""
        # 创建包含检索工具的 Agent
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[search_relevant_documents_from_knowledge_base_tool],
        )

        # 准备测试查询
        test_query = "灵格若产品"

        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": f"请帮我搜索关于 '{test_query}' 的相关信息"}
            ],
            "retriever_config": self.retriever_config,
            "context_tokens_limit": self.context_tokens_limit,
        },
            config={
                "recursion_limit": AGENT_MAX_ITERATIONS,
            },
            context=self.application_context
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        messages = result["messages"]
        self.assertGreater(len(messages), 0)

        # 打印检索结果
        print(f"\n{'='*60}")
        print(f"测试查询: {test_query}")
        print(f"{'='*60}")

        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n消息内容：{message.content[:500]}...")

            # 检查是否有 artifact（检索工具的元数据）
            if hasattr(message, 'artifact') and message.artifact:
                artifact = message.artifact
                print(f"\n检索元数据：")
                print(f"  - 文档数量: {artifact.get('retriever_docs_count', 0)}")
                print(f"  - 总 tokens: {artifact.get('retriever_docs_total_tokens', 0)}")
                print(f"  - 搜索关键词: {artifact.get('search_keywords', '')}")

    def test_search_relevant_documents_from_uploaded_files_with_agent(self):
        """测试使用 Agent 调用上传文件检索工具搜索相关文档"""
        # 创建包含上传文件检索工具的 Agent
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[search_relevant_documents_from_uploaded_files_tool],
        )

        uploaded_files_info = {
            "temp-ac403759-1fe6-4fb7-a804-86b6f2c2" : ChatSessionUploadFileInfo(
                chat_session_id="chat-session-test-1",
                file_sync_record_id="temp-ac403759-1fe6-4fb7-a804-86b6f2c2",
                file_name="FinDKG.md"
            )
        }
        chat_session_state = ChatSessionState(
            uploaded_files_info=uploaded_files_info
        )

        retriever_config = RetrieverConfig(
            retriever_type="A01",
            knowledge_index_folder_id='af60402d',
            knowledge_index_folder_name='af60402d/1761379771990',
            version='1761379771990'
        )

        # 准备测试查询
        test_query = "能介绍下文档提到的Dynamic knowledge graph是什么？"

        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": f"{test_query}"}
            ],
            "retriever_config": retriever_config,
            "context_tokens_limit": self.context_tokens_limit,
            "chat_session_state": chat_session_state,
        },
            config={
                "recursion_limit": AGENT_MAX_ITERATIONS,
            },
            context=self.application_context
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        messages = result["messages"]
        self.assertGreater(len(messages), 0)

        # 打印检索结果
        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n消息内容：{message.content}")

            # 检查是否有 artifact（检索工具的元数据）
            if hasattr(message, 'artifact') and message.artifact:
                artifact = message.artifact
                print(f"\n检索元数据：")
                print(f"  - 文档数量: {artifact.get('retriever_docs_count', 0)}")
                print(f"  - 总 tokens: {artifact.get('retriever_docs_total_tokens', 0)}")
                print(f"  - 搜索关键词: {artifact.get('search_keywords', '')}")

    def test_search_uploaded_files_no_files_uploaded(self):
        """测试当没有上传文件时的情况"""
        # 创建空的聊天会话状态
        empty_chat_session_state = ChatSessionState(
            uploaded_files_info={}
        )

        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[search_relevant_documents_from_uploaded_files_tool],
        )

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": "请搜索上传文件中的信息"}
            ],
            "retriever_config": self.retriever_config,
            "context_tokens_limit": self.context_tokens_limit,
            "chat_session_state": empty_chat_session_state,
        },
            config={
                "recursion_limit": AGENT_MAX_ITERATIONS,
            },
            context=self.application_context
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)

        print(f"\n{'='*60}")
        print(f"测试场景: 没有上传文件")
        print(f"{'='*60}")

        for message in result["messages"]:
            if hasattr(message, 'content') and message.content:
                print(f"\n消息内容：{message.content}")
                # 验证返回了预期的提示信息
                if "No files have been uploaded" in message.content:
                    print("✓ 正确处理了无上传文件的情况")
