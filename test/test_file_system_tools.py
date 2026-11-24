import unittest

from langchain.agents import create_agent

from configs.model_config import AGENT_MAX_ITERATIONS
from context.context import ApplicationContext
from context.state import ChatState
from controllers.vo.startup_llm_model_request_vo import ChatSessionState, ChatSessionUploadFileInfo, RetrieverConfig
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from tools_definitions.tools.file_system_tools import read_file_content_from_knowledge_tool, \
    read_uploaded_file_content_tool


class TestFileSystemTools(unittest.TestCase):
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
        self.context_tokens_limit = 20000

    def test_read_file_content_from_knowledge_tool_with_agent(self):
        """测试使用 Agent 调用知识库文件读取工具"""
        # 创建包含文件读取工具的 Agent
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[read_file_content_from_knowledge_tool],
        )

        # 准备测试参数 - 使用一个已知的文件 sync_record_id
        test_file_sync_record_id = "temp-ac403759-1fe6-4fb7-a804-86b6f2c2"
        test_query = f"请读取文件 {test_file_sync_record_id} 的内容，然后做一个简要的总结"

        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": test_query}
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

        # 打印读取结果
        print(f"\n{'='*60}")
        print(f"测试：读取知识库文件内容")
        print(f"文件 ID: {test_file_sync_record_id}")
        print(f"{'='*60}")

        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n消息内容：{message.content[:500]}...")
                continue

            # 检查是否有 artifact（文件读取工具的元数据）
            if hasattr(message, 'artifact') and message.artifact:
                artifact = message.artifact
                print(f"\n文件读取元数据：")
                print(f"  - 文件名: {artifact.get('file_name', 'N/A')}")
                print(f"  - 当前页: {artifact.get('current_page', 0)}")
                print(f"  - 页面 tokens: {artifact.get('page_tokens', 0)}")
                print(f"  - 总 tokens: {artifact.get('total_tokens', 0)}")
                print(f"  - 是否有下一页: {artifact.get('has_next_page', False)}")
                print(f"  - 分页模式: {artifact.get('pagination_mode', 'N/A')}")

                # 验证关键字段存在
                self.assertIn('file_name', artifact)
                self.assertIn('current_page', artifact)
                self.assertIn('page_tokens', artifact)
                self.assertIn('total_tokens', artifact)
                self.assertIn('has_next_page', artifact)
                continue

            print(f"\n AI message:{message}")



    def test_read_uploaded_file_content_tool_with_agent(self):
        """测试使用 Agent 调用上传文件读取工具"""
        # 创建包含上传文件读取工具的 Agent
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[read_uploaded_file_content_tool],
        )

        # 准备上传文件信息
        uploaded_files_info = {
            "extension-mjdrdyb4x1glxcfrc": ChatSessionUploadFileInfo(
                chat_session_id="session-0hmJnW",
                file_sync_record_id="extension-mjdrdyb4x1glxcfrc",
                file_name="Built-in middleware - Docs by .md"
            )
        }
        chat_session_state = ChatSessionState(
            uploaded_files_info=uploaded_files_info
        )

        # 准备测试查询
        test_query = "请读取我上传的文件内容，然后做一个简要的总结"

        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": test_query}
            ],
            "retriever_config": self.retriever_config,
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

        # 打印读取结果
        print(f"\n{'='*60}")
        print(f"测试：读取上传文件内容")
        print(f"{'='*60}")

        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n消息内容：{message.content[:500]}...")
                continue

            # 检查是否有 artifact（文件读取工具的元数据）
            if hasattr(message, 'artifact') and message.artifact:
                artifact = message.artifact
                print(f"\n文件读取元数据：")
                print(f"  - 文件名: {artifact.get('file_name', 'N/A')}")
                print(f"  - 文件 ID: {artifact.get('file_sync_record_id', 'N/A')}")
                print(f"  - 当前页: {artifact.get('current_page', 0)}")
                print(f"  - 页面 tokens: {artifact.get('page_tokens', 0)}")
                print(f"  - 总 tokens: {artifact.get('total_tokens', 0)}")
                print(f"  - 是否有下一页: {artifact.get('has_next_page', False)}")
                print(f"  - 分页模式: {artifact.get('pagination_mode', 'N/A')}")

                # 验证关键字段存在
                self.assertIn('file_name', artifact)
                self.assertIn('file_sync_record_id', artifact)
                self.assertIn('current_page', artifact)
                self.assertIn('page_tokens', artifact)
                self.assertIn('total_tokens', artifact)
                self.assertIn('has_next_page', artifact)
                continue

            print(f"\n AI message:{message}")



    def test_read_uploaded_file_multiple_pages(self):
        """测试读取上传文件的多个页面"""
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[read_uploaded_file_content_tool],
        )

        uploaded_files_info = {
            "temp-ac403759-1fe6-4fb7-a804-86b6f2c2": ChatSessionUploadFileInfo(
                chat_session_id="chat-session-test-1",
                file_sync_record_id="temp-ac403759-1fe6-4fb7-a804-86b6f2c2",
                file_name="FinDKG.md"
            )
        }
        chat_session_state = ChatSessionState(
            uploaded_files_info=uploaded_files_info
        )

        print(f"\n{'='*60}")
        print(f"测试：分页读取上传文件")
        print(f"{'='*60}")

        # 读取第一页
        for page_num in [1, 2]:
            test_query = f"请读取我上传的文件的第{page_num}页内容，每页最多2000个tokens"

            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": test_query}
                ],
                "retriever_config": self.retriever_config,
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
            print(f"\n--- 第 {page_num} 页 ---")

            for message in result["messages"]:
                if hasattr(message, 'artifact') and message.artifact:
                    artifact = message.artifact
                    print(f"当前页: {artifact.get('current_page')}")
                    print(f"页面 tokens: {artifact.get('page_tokens')}")
                    print(f"是否有下一页: {artifact.get('has_next_page')}")

                    # 验证页码正确
                    self.assertEqual(artifact.get('current_page'), page_num)

            # 如果没有下一页，停止测试
            has_next = False
            for message in result["messages"]:
                if hasattr(message, 'artifact') and message.artifact:
                    has_next = message.artifact.get('has_next_page', False)
                    break

            if not has_next:
                print(f"\n✓ 已读取到最后一页（第 {page_num} 页）")
                break


if __name__ == '__main__':
    unittest.main()

