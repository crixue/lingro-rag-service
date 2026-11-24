import os
import unittest

from langchain.agents import create_agent

from configs.model_config import BASE_DIR, AGENT_MAX_ITERATIONS
from context.context import ApplicationContext
from context.state import ChatState
from controllers.vo.startup_llm_model_request_vo import RetrieverConfig, ChatSessionState
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from tools_definitions.tools.knowledge_base_info_tools import get_knowledge_base_files_detail_tool
from tools_definitions.tools.summarize_tools import summarize_knowledge_base_file_tool, summarize_text_tool, \
    summary_text, summarize_chat_session_uploaded_file_tool
from utils.common_util import count_tokens


class TestSummaryTools(unittest.TestCase):

    def setUp(self):
        self.analysis_result_folder_name = "b78a01e8/1761547417618"
        self.user_id = "92040b49"
        self.document_path_list = [
            ("doc_001", os.path.join(BASE_DIR, "samples", "analysis_docs", "FinDKG.md")),
            ("doc_002", os.path.join(BASE_DIR, "samples", "analysis_docs", "1-words-cn.txt"))
        ]
        model_config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "streaming": True,
            "max_tokens": 2000,
        }
        self.chat_model = DefaultLangchainLLM.llm_client_factory(
            product_id="MODEL-0008",
            llm_model_name="qwen-flash",
            user_id=self.user_id,
            llm_model_config=model_config,
            additional_llm_config={"parallel_tool_calls": False}
        )

        # 创建用户上下文
        self.application_context = ApplicationContext(
            user_id=self.user_id)
        self.retriever_config: RetrieverConfig = RetrieverConfig(
            is_chinese_text=False,
            retriever_type="A01",
            knowledge_index_folder_id='af60402d',
            version='1761379771990'
        )
        self.chat_session_state: ChatSessionState = ChatSessionState(
            uploaded_files_info = {
                "mh6a08rojdfapdi98" : {
                    "file_name": "60公里长的宇宙飞船“蝶蛹号” 带人类单程前往半人马座-1761396640573.md",
                    "file_size": 12062,
                    "last_modified": "2024-05-24 10:16:11",
                    "chat_session_id": "chat-session-test-1"
                }
            }
        )

    def test_count_tokens(self):
        file_path = self.document_path_list[0][1]
        file_load1 = DefaultFileLoaderComponent(file_path=file_path, is_chinese_text=False)
        content1 = file_load1.load_full_content()
        print(count_tokens(content1.page_content))  # 14297 tokens

        file_path1 = self.document_path_list[1][1]
        file_load2 = DefaultFileLoaderComponent(file_path=file_path1, is_chinese_text=True)
        content2 = file_load2.load_full_content()
        print(count_tokens(content2.page_content))  # 1434 tokens

    def test_summarize_file_tool_with_agent(self):
        """测试使用 Agent 调用 summarize_file_tool 摘要文件"""

        self.chat_model.bind_tools([get_knowledge_base_files_detail_tool, summarize_knowledge_base_file_tool],
                                   parallel_tool_calls=False)
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[get_knowledge_base_files_detail_tool, summarize_knowledge_base_file_tool],
        )

        # 准备测试文件路径
        test_file_path = '60公里长的宇宙飞船“蝶蛹号” 带人类单-1761382363801.md'

        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": f"请帮我总结下 {test_file_path} 这个文件的内容"}
            ],
            "retriever_config": self.retriever_config
        },
            config={
                "recursion_limit": AGENT_MAX_ITERATIONS,
            },
            context=self.application_context)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        messages = result["messages"]
        self.assertGreater(len(messages), 0)

        # 打印摘要结果
        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n摘要结果：{message.content}")

    def test_summarize_file_tool_with_agent1(self):
        """测试使用 Agent 调用 summarize_file_tool 摘要文件"""
        # 创建包含文件摘要工具的 Agent
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[get_knowledge_base_files_detail_tool, summarize_knowledge_base_file_tool],
        )

        # 准备测试文件路径
        # test_file_path = self.document_path_list[0][1]  # 使用较大的中文文件
        test_file_path = os.path.join(BASE_DIR, "samples", "analysis_docs", "FlashAttention.md")
        # 调用 Agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": f"请帮我总结这个文件的内容：{test_file_path}"}
            ]
        }, context=self.application_context)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        messages = result["messages"]
        self.assertGreater(len(messages), 0)

        # 打印摘要结果
        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n摘要结果：{message.content}")

    def test_summarize_text_tool_with_agent(self):
        """测试使用 Agent 调用 summarize_text_tool 摘要文本"""
        # 创建包含文本摘要工具的 Agent
        file_path1 = os.path.join(BASE_DIR, "samples", "analysis_docs", "HNHW graph.md")
        # 准备测试文本
        file_load = DefaultFileLoaderComponent(file_path=file_path1, is_chinese_text=True)
        test_text = file_load.load_full_content()
        summary_result = summary_text(text=test_text.page_content, user_id=self.user_id,
                                      reduce_phase_template_without_metadata=True)
        print(f"\n直接调用 summary_text 函数得到的摘要结果：{summary_result}")

    def test_summarize_text_tool_with_empty_text(self):
        """测试使用 Agent 处理空文本"""
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            tools=[summarize_text_tool],
        )

        empty_text = "   "

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": f"请总结这段文本：{empty_text}"}
            ]
        }, context=self.application_context)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)

        # 打印结果
        for message in result["messages"]:
            if hasattr(message, 'content'):
                print(f"\n空文本处理结果：{message.content}")

    def test_summarize_chat_session_file_tool_with_agent(self):
        """测试使用 Agent 调用 summarize_chat_session_file_tool 摘要聊天会话中上传的文件"""
        agent = create_agent(
            model=self.chat_model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=[summarize_chat_session_uploaded_file_tool],
        )

        # 调用 Agent，请求总结聊天会话中上传的文件
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": "请帮我总结一下我上传的文件内容"}
            ],
            "chat_session_state": self.chat_session_state
        },
            config={
                "recursion_limit": AGENT_MAX_ITERATIONS,
            },
            context=self.application_context)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        messages = result["messages"]
        self.assertGreater(len(messages), 0)

        # 打印摘要结果
        for message in messages:
            if hasattr(message, 'content') and message.content:
                print(f"\n聊天会话文件摘要结果：{message.content}")

