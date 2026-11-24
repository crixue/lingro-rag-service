import asyncio
import json
import os
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Union, Optional

import requests
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from configs.model_config import logger, AGENT_MAX_ITERATIONS, JAVA_MICRO_SERVICE_URL
from controllers.vo.startup_llm_model_request_vo import ChatMessage, RetrieverConfig, \
    UsedModelConfig, ChatRequestVO, ChatResponseVO
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from prompt_definitions import PromptDefinition
from rag_checker.container import RAGResult, RetrievedDoc, RAGResults
from rag_checker.evaluator import RAGChecker
from service.chat_agent_service import DefaultChatAgentService
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from utils.lang_util import lang_def_name


class RAGTestItem(BaseModel):
    query_id: str = None
    query: str = None
    gt_answer: str = Field(default=None, description="问题的标准答案")
    response: str = Field(default=None, description="RAG系统基于问题和检索上下文生成的最终输出")
    level: Optional[str] = Field(default=None, description="问题的难度等级", examples=["easy", "medium", "hard"])
    retrieved_context: Optional[list[Document]] = Field(default=None, description="检索到的上下文", )


class RAGEvaluator:
    def __init__(self,
                 is_chinese_text: bool = True,
                 user_id: str = None,
                 product_id: str = None,
                 rating_product_id: str = None,
                 **kwargs):
        assert user_id, "The user_id should be provided."
        assert product_id, "The product_id should be provided."
        self.is_chinese_text = is_chinese_text
        self.user_id = user_id
        self.product_id = product_id
        self.system_prompt = None
        self.default_agent_service = None
        self.model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=product_id)
        self.model_name = self.model_info.get("name")
        self.rating_product_id = rating_product_id
        self.rating_model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=rating_product_id)

    async def read_corpus_file_and_evaluate_async(self, *, test_dataset_file_path: str,
                                                  saved_result_dir: str,
                                                  retriever_config: RetrieverConfig = None,
                                                  used_model_config: UsedModelConfig = None,
                                                  **kwargs):
        assert test_dataset_file_path.endswith(".json") or test_dataset_file_path.endswith(
            ".jsonl"), "The file should be a json file."
        assert saved_result_dir, "The saved_result_dir should be provided."
        assert retriever_config, "The retriever_config should be provided."
        assert used_model_config, "The init_model_request should be provided."
        only_evaluate_rag_results = kwargs.get("only_evaluate_rag_results", False)
        only_answer_questions = kwargs.get("only_answer_questions", False)
        assert not (
                    only_evaluate_rag_results and only_answer_questions), "only_evaluate_rag_results and only_answer_questions should not be True at the same time."

        if test_dataset_file_path.endswith(".json"):
            with open(test_dataset_file_path, "r", encoding="utf-8") as f:
                json_list = json.load(f)
        elif test_dataset_file_path.endswith(".jsonl"):
            with open(test_dataset_file_path, "r", encoding="utf-8") as f:
                json_list = [json.loads(line) for line in f]
        test_file_name = test_dataset_file_path.split("/")[-1].split(".")[0]
        if (retriever_type := retriever_config.retriever_type) is None:
            retriever_type = "default"
        saved_result_file_path = f"{saved_result_dir}/{self.model_name}_{retriever_type}_{test_file_name}_result.jsonl"

        if not only_evaluate_rag_results:
            batch_data = [RAGTestItem(**item) for item in json_list if item.get("query") and item.get("gt_answer")]
            if len(batch_data) == 0:
                raise ValueError(f"No valid data found in the file:{test_dataset_file_path}")
            if kwargs.get("test_count", None):
                skip_count = kwargs.get("skip_count", 30)
                batch_data = batch_data[skip_count: skip_count + kwargs.get("test_count")]
            retriever_config.is_chinese_text = self.is_chinese_text
            used_model_config.is_chinese_text = self.is_chinese_text
            # 如果存在结果文件，先删除
            if os.path.exists(saved_result_file_path):
                os.remove(saved_result_file_path)
            await self._batch_answer_questions_async(batch_data, saved_result_file_path, retriever_config,
                                                     used_model_config, **kwargs)
            logger.info(
                f"[read_corpus_file_and_evaluate_async]-Evaluate result has been saved to {saved_result_file_path}")
        if not only_answer_questions:
            self.batch_evaluate_rag_results(saved_result_file_path, saved_result_dir, **kwargs)

    def single_request_completion(self, chat_request: ChatRequestVO, langchain_llm: DefaultLangchainLLM,
                                  agent_max_iterations: int = AGENT_MAX_ITERATIONS):
        default_agent_service = DefaultChatAgentService(langchain_llm=langchain_llm, react_system_prompt="")
        return default_agent_service.completions(chat_request, agent_max_iterations=agent_max_iterations)

    async def _batch_answer_questions_async(self, batch_data: list[RAGTestItem], file_path: str,
                                            retriever_config: RetrieverConfig = None,
                                            used_model_config: UsedModelConfig = None, **kwargs):
        agent_max_iterations = kwargs.get("agent_max_iterations", AGENT_MAX_ITERATIONS)
        # 查看file_path的dir是否存在，不存在则创建
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 提供可供选择的系统提示，默认使用icot_prompt
        system_prompt_str = kwargs.get("system_prompt", None)
        if system_prompt_str is None:
            icot_system_prompt = PromptDefinition.load_builtin_prompt_static(lang=lang_def_name(self.is_chinese_text),
                                                                             prompt_name="icot_prompt")
            few_shot_samples = kwargs.get("few_shot_samples")
            if few_shot_samples:
                if isinstance(few_shot_samples, list):
                    few_shot_samples = "\n".join(few_shot_samples)
                elif isinstance(few_shot_samples, str):
                    few_shot_samples = few_shot_samples
            else:
                few_shot_samples = ""
            system_prompt_str = icot_system_prompt.format(
                agent_max_iterations=agent_max_iterations,
                few_shot_samples=few_shot_samples,
            )
        history_messages = [ChatMessage(
            role="system",
            content=system_prompt_str
        )]
        max_concurrency = kwargs.get("max_concurrency", 8)
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for i in range(0, len(batch_data), max_concurrency):
                sub_chat_batch_requests = []
                sub_batch_data = {}
                for item in batch_data[i:i + max_concurrency]:
                    id = item.query_id if item.query_id else str(uuid.uuid4())
                    sub_batch_data[id] = item
                    sub_chat_batch_requests.append(ChatRequestVO(
                        id=id,
                        query=item.query,
                        history_messages=history_messages,
                        is_chinese_text=self.is_chinese_text,
                        retriever_config=retriever_config,
                        used_model_config=used_model_config
                    ))
                langchain_llm = DefaultLangchainLLM(
                    model_name=used_model_config.llm_model_name,
                    model_config={
                        "temperature": 0.5,
                        "streaming": False,
                        "repetition_penalty": 1.2
                    },
                    product_id=self.product_id,
                    user_id=self.user_id,
                )

                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(executor, self.single_request_completion, item, langchain_llm,
                                              agent_max_iterations)
                         for item in sub_chat_batch_requests]
                results = await asyncio.gather(*tasks)
                # results = []
                # for item in sub_chat_batch_requests:
                #     result = self.single_request_completion(item, langchain_llm, agent_max_iterations)
                #     results.append(result)
                with open(file_path, "a", encoding="utf-8") as f:
                    for future in results:
                        try:
                            chat_resp: ChatResponseVO = future
                            id = chat_resp.id
                            if not id or id not in sub_batch_data:
                                logger.warning(
                                    f"[_batch_evaluate_questions_async]-chat_resp.id is None or not found in sub_batch_data, just skip")
                                continue
                            ai_message = chat_resp.choices[0].message.content
                            test_item: RAGTestItem = sub_batch_data[id]
                            test_item.response = ai_message
                            if chat_resp.retriever_docs:
                                test_item.retrieved_context = chat_resp.retriever_docs
                            else:
                                test_item.retrieved_context = []
                            logger.debug(
                                f"[_batch_evaluate_questions_async]-Evaluate result id:{id} question:{test_item.query} and response:{ai_message}")
                            f.write(json.dumps(test_item.model_dump(), ensure_ascii=False) + "\n")
                        except Exception as exc:
                            logger.error(f"[_batch_evaluate_questions_async] process throw an exception:{exc}")
                            pass
                    logger.info(f"[_batch_evaluate_questions_async]-Evaluate result has been saved to {file_path}")

    def _custom_llm_api_func(self, prompts: list[str], **kwargs) -> list[str]:
        langchain_llm = DefaultLangchainLLM(
            model_name=self.rating_model_info.get("name"),
            model_config={
                "temperature": 0.1,
                # "top_p": 0.9,
                "streaming": False,
                "repetition_penalty": 1.2
            },
            product_id=self.rating_product_id,
            user_id=self.user_id,
        )
        response_list = prompts.copy()
        for index, message in langchain_llm.get_llm_client().batch_as_completed(inputs=prompts,
                                                                                config={
                                                                                    "max_concurrency": kwargs.get(
                                                                                        "max_concurrency", 8),
                                                                                    "callbacks": [
                                                                                        langchain_llm.llm_callback_client]
                                                                                }):
            if isinstance(message, AIMessage):
                response_list[index] = message.content
                query = prompts[index]
                logger.debug(f"[_custom_llm_api_func]-AI response for query:{query} and answer is:\n {message.content}")
            else:
                logger.warning(f"[_custom_llm_api_func]-Not AI message: {message}")

        return response_list

    def batch_evaluate_rag_results(self,
                                   rag_results_file_path: Union[str | LiteralString],
                                   saved_result_dir: Union[str | LiteralString],
                                   metrics_type: str = "overall_metrics",
                                   **kwargs):
        """
            Evaluate the RAG results in batch.
            :param rag_results_file_path: The file path of the RAG results. Generated by read_corpus_file_and_evaluate_async and should be a jsonl file.
            :param saved_result_dir: The directory to save the evaluation result.
            :param metrics_type: The metrics type to evaluate. Default is "overall_metrics".
                                Options are "overall_metrics","all_metrics","generator_metrics","retriever_metrics".
        """
        assert rag_results_file_path.endswith(".jsonl"), "The file should be a jsonl file."
        assert os.path.exists(rag_results_file_path), f"The file:{rag_results_file_path} should exist."
        assert saved_result_dir, "The saved_result_dir should be provided."

        task_id = kwargs.get("task_id", None)
        result = {
            "taskId": task_id,
            "status": 100,
        }

        evaluate_result_file_path = f"{saved_result_dir}/rag-checker_{task_id}_evaluate_{metrics_type}_result.json"
        evaluator = RAGChecker(
            extractor_max_new_tokens=kwargs.get("extractor_max_new_tokens", 1000),
            extractor_api_base=None,
            checker_api_base=None,
            batch_size_extractor=16,
            batch_size_checker=16,
            joint_check=False,
            custom_llm_api_func=self._custom_llm_api_func,
        )
        test_batch_count = kwargs.get("test_count", None)
        rag_result_list = []
        retrieved_context: List[RetrievedDoc] = []
        with open(rag_results_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if test_batch_count is not None and i >= int(test_batch_count):
                    break
                try:
                    item = RAGTestItem.model_validate_json(line)
                    if item.retrieved_context and len(item.retrieved_context) > 0:
                        for doc in item.retrieved_context:
                            if doc.page_content:
                                retrieved_item = RetrievedDoc(
                                    doc_id=doc.id,
                                    text=doc.page_content,
                                )
                                retrieved_context.append(retrieved_item)
                    if item.query and item.gt_answer and item.response:
                        result_item: RAGResult = RAGResult(
                            query_id=item.query_id,
                            query=item.query,
                            gt_answer=item.gt_answer,
                            response=item.response,
                            retrieved_context=retrieved_context
                        )
                        rag_result_list.append(result_item)
                except Exception as exc:
                    result["status"] = -100
                    logger.error(f"[batch_evaluate_rag_results]-Error in parsing line:{line}, with exception:{exc}")
        metrics: dict = evaluator.evaluate(RAGResults(results=rag_result_list), metrics_type,
                                           save_path=evaluate_result_file_path)
        result["resultFilepath"] = evaluate_result_file_path
        logger.info(f"[batch_evaluate_rag_results]-Evaluate result has been saved to {evaluate_result_file_path}")

        if result.get("taskId", None) is not None:
            try:
                requests.put(url=JAVA_MICRO_SERVICE_URL["notice-rag-check-result"],
                             headers={"Content-Type": "application/json"},
                             json=result)
                logger.info(
                    f"[batch_evaluate_rag_results]-Notice core-service the knowledge_index_folder_id-{kwargs.get('knowledge_index_folder_id', None)} rag result file path-{evaluate_result_file_path} have build complete")
            except Exception as e:
                logger.exception(
                    f"[batch_evaluate_rag_results]-Failed to notice core-service the knowledge_index_folder_id-{kwargs.get('knowledge_index_folder_id', None)} rag result file path-{evaluate_result_file_path} with ex: {e}")
        return metrics
