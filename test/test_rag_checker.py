import os
import unittest

from langchain_core.messages import AIMessage

from configs.model_config import KB_ROOT_PATH, RAG_CHECKER_DIR_NAME
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from rag_checker.rag_evaluate import RAGEvaluator


class TestRagChecker(unittest.TestCase):
    def test_rag_checker(self):
        def custom_llm_api_func(prompts: list[str], **kwargs) -> list[str]:
            """
            Get responses from LLM for the input prompts
            Parameters
            ----------
            prompts: List[str]
            A list of prompts.

            Returns
            ----------
            response_list : List[str]
                A list of generated text.
            """
            response_list = []
            langchain_llm = DefaultLangchainLLM(
                model_name="qwen-max",
                model_config={
                    "temperature": 0,
                    "top_p": 0.9,
                    "streaming": False,
                    "repetition_penalty": 1.2
                },
                product_id="MODEL-0004",
                user_id="92040b49",
            )
            messages = (langchain_llm.get_llm_client()
                        .batch_as_completed(inputs=prompts,
                                            config={"max_concurrency": kwargs.get("max_concurrency", 8),
                                                    "callbacks": [langchain_llm.llm_callback_client]}))
            for index, message in messages:
                if isinstance(message, AIMessage):
                    response_list.insert(index, message.content)
                else:
                    print(f"Not AI message: {message}")
            return response_list

        questions = ["What is the capital of France?", "What is the capital of Germany?"]
        responses = custom_llm_api_func(questions)
        print(responses)

    def test_rag_evaluate_rag_results(self):
        user_folder_id = "e9a3d019"
        local_vs_folder_name = "test_small_batch1"
        saved_result_dir = os.path.join(KB_ROOT_PATH, user_folder_id,
                                        local_vs_folder_name,
                                        RAG_CHECKER_DIR_NAME)
        rag_results_file_path = os.path.join(saved_result_dir, "hotpotqa_train_result.jsonl")
        test_count = 10
        is_chinese_text = False
        user_id = "92040b49"
        product_id = "MODEL-0004"
        metrics_type = "retriever_metrics"
        rag_evaluator = RAGEvaluator(is_chinese_text, user_id=user_id, product_id=product_id)
        rag_evaluator.batch_evaluate_rag_results(rag_results_file_path, saved_result_dir,
                                                 metrics_type=metrics_type,
                                                 test_count=test_count
                                                 )
