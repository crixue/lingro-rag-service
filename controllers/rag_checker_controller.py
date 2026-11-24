import json
import os

from configs.model_config import RAG_CHECKER_DIR_NAME, CACHE_ROOT_PATH
from controllers.vo.startup_llm_model_request_vo import RetrieverConfig, UsedModelConfig
from rag_checker.rag_evaluate import RAGEvaluator
from core.model_io.support_models_info_util import SupportedModelsInfoUtil


class RAGCheckerController:
    def __init__(self,
                 is_chinese_text: bool = True,
                 user_id: str = None,
                 product_id: str = None,
                 rating_product_id: str = None,
                 **kwargs):
        self.is_chinese_text = is_chinese_text
        if rating_product_id is None:
            rating_product_id = product_id
        self.rag_evaluator = RAGEvaluator(is_chinese_text, user_id=user_id, product_id=product_id, rating_product_id=rating_product_id)

    async def rag_evaluate_async(self,
                                 used_model_config: UsedModelConfig,
                                 retriever_config: RetrieverConfig,
                                 user_id: str = None,
                                 **kwargs):
        if not (saved_result_dir := kwargs.get("saved_result_dir", None)):
            saved_result_dir = os.path.join(CACHE_ROOT_PATH, retriever_config.knowledge_index_folder_id, RAG_CHECKER_DIR_NAME)
            if not os.path.exists(saved_result_dir):
                os.makedirs(saved_result_dir)
        kwargs.pop("saved_result_dir", None)
        if not (test_dataset_file_path := kwargs.get("test_dataset_file_path", None)):
            test_dataset_file_path = os.path.join(saved_result_dir, "rag_test_data.json")
            if not os.path.exists(test_dataset_file_path):
                #创建一个测试数据文件
                test_data = {}
                with open(test_dataset_file_path, "w") as f:
                    json.dump(test_data, f, ensure_ascii=False)
        kwargs.pop("test_dataset_file_path", None)

        model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=used_model_config.llm_id)
        if used_model_config.llm_id is None:
            used_model_config.llm_id = model_info.get("id")
        used_model_config.llm_model_name = model_info.get("name")
        model_config = {
                    "temperature": 0,
                    # "top_p": 0.9,
                    "streaming": False,
                    "repetition_penalty": 1.2
                }
        used_model_config.llm_model_config = model_config

        await self.rag_evaluator.read_corpus_file_and_evaluate_async(test_dataset_file_path=test_dataset_file_path,
                                                                     saved_result_dir=saved_result_dir,
                                                                     retriever_config=retriever_config,
                                                                     used_model_config=used_model_config,
                                                                     **kwargs)