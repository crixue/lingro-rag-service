import os
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from configs.model_config import BASE_DIR
from core.data_connections.retrievers.document_analysis_retriever.global_document_analysis import GlobalDocumentAnalysis
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM


class TestGlobalDocumentAnalysis(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.analysis_result_folder_name = "b78a01e8/1761547417618"
        self.user_id = "92040b49"
        self.document_path_list = [
            ("doc_001", os.path.join(BASE_DIR, "samples", "analysis_docs", "1-words-cn.txt")),
            ("doc_002", os.path.join(BASE_DIR, "samples", "analysis_docs", "2-words-cn.txt"))
        ]
        model_config = {
                "temperature": 0.5,
                "top_p": 0.9,
                "streaming": True,
                "max_tokens": 2000
            }
        langchain_llm = DefaultLangchainLLM(
            model_name="glm-4-flash-250414",
            user_id="7af04ac6",
            product_id="MODEL-0205",
            model_config=model_config)

    async def test_analysis_documents_and_save_async(self):
        """测试文档分析和保存功能"""
        # 创建 GlobalDocumentAnalysis 实例
        analysis = GlobalDocumentAnalysis(
            analysis_result_folder_name=self.analysis_result_folder_name,
            is_chinese_text=True
        )

        # 执行文档分析
        await analysis.analysis_documents_and_save_async(
            user_id=self.user_id,
            document_path_list=self.document_path_list,
            require_analysis=True
        )

    # def test_retrieve_analysis_result_documents(self):
    #     analysis = GlobalDocumentAnalysis(
    #         analysis_result_folder_name=self.analysis_result_folder_name,
    #         is_chinese_text=True
    #     )
    #     docs = analysis.retrieve_analysis_result_documents(ids=["doc_002"])
    #     for doc in docs:
    #         print(doc.model_dump())
    #     self.assertTrue(len(docs) > 0, "No documents were saved in the storage.")


    async def test_analysis_documents_and_save_async_with_mock(self):
        """使用 mock 测试文档分析和保存功能"""
        with patch('core.data_connections.retrievers.document_analysis.global_document_analysis.DefaultLangchainLLM') as mock_llm, \
             patch('core.data_connections.retrievers.document_analysis.global_document_analysis.create_agent') as mock_create_agent:

            # Mock LLM 和 agent
            mock_model = MagicMock()
            mock_llm.llm_client_factory.return_value = mock_model

            mock_agent = AsyncMock()
            mock_create_agent.return_value = mock_agent

            # Mock agent 返回结果
            from langchain_core.messages import AIMessage
            mock_agent.abatch_as_completed.return_value = [
                {
                    "messages": [AIMessage(content='{"summary": "测试摘要", "keywords": ["测试"]}')]
                }
            ]

            # 创建实例并测试
            analysis = GlobalDocumentAnalysis(
                analysis_result_folder_name=self.analysis_result_folder_name,
                is_chinese_text=True
            )

            result = await analysis.analysis_documents_and_save_async(
                user_id=self.user_id,
                document_path_list=self.document_path_list,
                require_analysis=True
            )

            # 验证调用
            mock_llm.llm_client_factory.assert_called_once()
            mock_create_agent.assert_called_once()


    def tearDown(self):
        """清理测试数据"""
        # 可以在这里清理测试过程中创建的文件和目录
        pass


if __name__ == '__main__':
    unittest.main()
