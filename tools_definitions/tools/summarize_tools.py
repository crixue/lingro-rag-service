import json
import os
import requests
from typing import Annotated, Optional, Dict

from json_repair import repair_json
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import ToolException, InjectedToolArg
from pydantic import BaseModel, Field, ValidationError

from configs.model_config import SUMMARY_CHUNK_SIZE, logger, SUMMARY_MAP_ANALYSIS_PRODUCT_ID, \
    SUMMARY_MAP_ANALYSIS_MODEL_CONFIG, SUMMARY_REDUCE_ANALYSIS_PRODUCT_ID, SUMMARY_REDUCE_ANALYSIS_MODEL_CONFIG, \
    DOCUMENT_ANALYSIS_MODEL_PRODUCT_ID, DOCUMENT_ANALYSIS_MODEL_CONFIG, SUMMARY_CHUNK_OVERLAP_TOKENS, \
    DIRECT_SUMMARY_TOKEN_THRESHOLD, JAVA_MICRO_SERVICE_URL, KB_ROOT_PATH, CHAT_SESSION_TEMP_FILE_DIR_NAME
from controllers.common.common_constant import DOC_STORE_SUFFIX
from controllers.vo.startup_llm_model_request_vo import ChatSessionUploadFileInfo
from core.common.common_constant import DocumentAnalysisConstants
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.data_connections.document_loaders.file_loader_v2 import ContentStringLoader
from core.data_connections.retrievers.storage.documents_storage import GlobalDocumentContextStorage
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from json_schema_definitions import JsonSchemaDefinition
from prompt_definitions import PromptDefinition
from tools_definitions.lang import ToolSchemaDefinition
from tools_definitions.schemas.build_task_response import BuildTaskFileInfo
from tools_definitions.tools.tool_utils import get_build_task_file_detail
from utils.common_util import count_tokens
from utils.lang_util import lang_def_name


def _invoke_doc_map_analysis(text_list: list[str], user_id: str, **kwargs) -> list[tuple]:
    is_chinese_text = kwargs.get("is_chinese_text", True)
    default_system_prompt = PromptDefinition.load_prompt_json_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        prompt_name="analysis_document_system_prompt").get("map_phase_template", "")
    system_prompt = kwargs.get("map_system_prompt", default_system_prompt)
    model_id = kwargs.get("map_model_id", SUMMARY_MAP_ANALYSIS_PRODUCT_ID)
    model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=model_id)
    model = DefaultLangchainLLM.llm_client_factory(
        llm_model_name=model_info.get("name", "qwen-flash"),
        user_id=user_id,
        product_id=model_id,
        llm_model_config=kwargs.get("map_llm_model_config", SUMMARY_MAP_ANALYSIS_MODEL_CONFIG),
    )
    response_json_schema = JsonSchemaDefinition.load_builtin_json_schema_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        schema_name="analysis_document_json_schema")
    model.with_structured_output(response_json_schema,
                                 method="json_schema", )

    agent = create_agent(
        model=model,
    )
    inputs = []
    for text in text_list:
        system_message = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        user_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"{text}",
                }
            ]
        )
        inputs.append({"messages": [
            system_message,
            user_message
        ]})

    results: list[tuple] = []
    index = 0
    try:
        for result in agent.batch_as_completed(inputs,
                                               config={
                                                   'max_concurrency': 4,  # Limit to 4 parallel calls
                                               }):
            try:
                structured_output = result[-1].get("messages")[-1]
                if not isinstance(structured_output, AIMessage):
                    logger.warning(
                        f"[analysis_documents_and_save_async]-message is not AIMessage, result:{structured_output}")
                    continue
                json_result = repair_json(structured_output.content, ensure_ascii=False)
                results.append((index, json_result))
                index += 1
            except Exception as e1:
                logger.exception(f"[invoke_doc_map_analysis]-structured_output error: {e1}")
    except Exception as e:
        logger.exception(f"[invoke_doc_map_analysis]-create_agent error: {e}")
        raise e
    return results


def _invoke_aggregate_map_results(map_results: list[tuple], **kwargs) -> dict:
    combined_summary_result = {}
    combined_summary = ""
    combined_theme_tags = set()
    combined_key_entities = {}

    for index, result in map_results:
        if result and isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception as e:
                logger.warning(f"[invoke_aggregate_map_results]-json.loads error: {e}, result:{result}")
                continue

        core_summary = result.get(DocumentAnalysisConstants.CORE_SUMMARY, "")
        combined_summary += (
            f"<<<DOC {index + 1}/{len(map_results)}>>>\n"
            f"{core_summary}\n"
            f"<<<END DOC>>>\n\n"
        )
        theme_tags = result.get(DocumentAnalysisConstants.THEME_TAGS, [])
        if theme_tags:
            combined_theme_tags.update(theme_tags)

        key_entities = result.get(DocumentAnalysisConstants.KEY_ENTITIES, [])
        if key_entities:
            for entity in key_entities:
                name = entity.get("entity_name")
                if name and name not in combined_key_entities:
                    combined_key_entities[name] = entity

    combined_summary_result[DocumentAnalysisConstants.CORE_SUMMARY] = combined_summary.strip()
    combined_summary_result[DocumentAnalysisConstants.THEME_TAGS] = list(combined_theme_tags)
    combined_summary_result[DocumentAnalysisConstants.KEY_ENTITIES] = list(combined_key_entities.values())
    print(f"[invoke_aggregate_map_results]-combined_summary_result: {combined_summary_result}")
    return combined_summary_result


def _invoke_reduce_summary(map_summary_result: dict, user_id: str, **kwargs):
    is_chinese_text = kwargs.get("is_chinese_text", True)
    if kwargs.get("reduce_phase_template_without_metadata", True) is True:
        default_system_prompt = PromptDefinition.load_prompt_json_dict_static(
            lang=lang_def_name(is_chinese_text=is_chinese_text),
            prompt_name="analysis_document_system_prompt").get("reduce_phase_template_without_metadata", "")
    else:
        default_system_prompt = PromptDefinition.load_prompt_json_dict_static(
            lang=lang_def_name(is_chinese_text=is_chinese_text),
            prompt_name="analysis_document_system_prompt").get("reduce_phase_template", "")

    system_prompt = kwargs.get("reduce_system_prompt", default_system_prompt)
    model_id = kwargs.get("reduce_model_id", SUMMARY_REDUCE_ANALYSIS_PRODUCT_ID)  # 相比较map阶段，reduce阶段可以使用更强的模型
    model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=model_id)
    model = DefaultLangchainLLM.llm_client_factory(
        llm_model_name=model_info.get("name", "qwen-max"),
        user_id=user_id,
        product_id=model_id,
        llm_model_config=kwargs.get("reduce_llm_model_config", SUMMARY_REDUCE_ANALYSIS_MODEL_CONFIG),
    )
    response_json_schema = JsonSchemaDefinition.load_builtin_json_schema_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        schema_name="analysis_document_json_schema")
    model.with_structured_output(response_json_schema,
                                 method="json_schema", )
    agent = create_agent(
        model=model,
    )
    user_input_json = json.dumps(map_summary_result, ensure_ascii=False)
    system_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
    user_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"{user_input_json}",
            }
        ]
    )
    messages = [
        system_message,
        user_message
    ]
    try:
        result = agent.invoke({"messages": messages})
        structured_output = result.get("messages")[-1]
        if not isinstance(structured_output, AIMessage):
            logger.warning(f"[invoke_reduce_summary]-message is not AIMessage, result:{structured_output}")
            return ""
        json_result = repair_json(structured_output.content, ensure_ascii=False)
        logger.info(f"[invoke_reduce_summary]-reduce summary result: {json_result}")
        return json_result
    except Exception as e:
        logger.exception(f"[invoke_reduce_summary]-create_agent error: {e}")
        raise e


def map_reduce_summarize(chunks: list[str], user_id: str, **kwargs) -> str | None:
    try:
        map_results = _invoke_doc_map_analysis(chunks, user_id, **kwargs)
        aggregated_map_result = _invoke_aggregate_map_results(map_results)
        reduced_summary = _invoke_reduce_summary(aggregated_map_result, user_id, **kwargs)
        if reduced_summary and isinstance(reduced_summary, dict) and kwargs.get("record_map_phase_summary",
                                                                                False) is True:
            reduced_summary[DocumentAnalysisConstants.MAP_PHASE_SUMMARY] = aggregated_map_result.get(
                DocumentAnalysisConstants.CORE_SUMMARY, "")
        return reduced_summary
    except Exception as e:
        exception_err_msg = f"[map_reduce_summarize]-map reduce summarize error: {e}"
        logger.exception(exception_err_msg)
        raise ToolException(exception_err_msg)


def direct_summary(text: str, user_id: str, **kwargs) -> str | None:
    is_chinese_text = kwargs.get("is_chinese_text", True)
    default_system_prompt = PromptDefinition.load_prompt_json_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        prompt_name="analysis_document_system_prompt").get("template", "")
    system_prompt = kwargs.get("system_prompt", default_system_prompt)

    model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=DOCUMENT_ANALYSIS_MODEL_PRODUCT_ID)
    model = DefaultLangchainLLM.llm_client_factory(
        llm_model_name=model_info.get("name", "qwen-flash"),
        user_id=user_id,
        product_id=DOCUMENT_ANALYSIS_MODEL_PRODUCT_ID,
        llm_model_config=DOCUMENT_ANALYSIS_MODEL_CONFIG,
    )
    response_json_schema = JsonSchemaDefinition.load_builtin_json_schema_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        schema_name="analysis_document_json_schema")
    model.with_structured_output(response_json_schema,
                                 method="json_schema", )
    agent = create_agent(
        model=model,
    )
    system_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
    user_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"{text}",
            }
        ]
    )
    messages = [
        system_message,
        user_message
    ]
    try:
        result = agent.invoke({"messages": messages})
        structured_output = result.get("messages")[-1]
        if not isinstance(structured_output, AIMessage):
            logger.warning(f"[direct_summary]-message is not AIMessage, result:{structured_output}")
            return ""
        json_result = repair_json(structured_output.content, ensure_ascii=False)
        logger.info(f"[direct_summary]-direct summary result: {json_result}")
        return json_result
    except Exception as e:
        logger.exception(f"[direct_summary]-create_agent error: {e}")
        raise e


def map_reduce_summarize_file_content(file_path: str, user_id: str, **kwargs) -> str | None:
    if not os.path.exists(file_path):
        logger.warning(f"[map_reduce_summarize_file_content]-document_path not exists: {file_path}")
        raise ToolException("The file does not exist or the path is invalid.")
    chunk_size = kwargs.get("summary_chunk_size", SUMMARY_CHUNK_SIZE)  # 每个块的大小，可以根据需要调整
    chunk_overlap = kwargs.get("summary_chunk_overlap", SUMMARY_CHUNK_OVERLAP_TOKENS)  # 块之间的重叠大小，可以根据需要调整
    file_loader = DefaultFileLoaderComponent(file_path=file_path, max_characters=chunk_size, overlap=chunk_overlap)
    docs = file_loader.load()
    chunks = [doc.page_content for doc in docs if doc.page_content]
    if not chunks:
        logger.warning(f"[map_reduce_summarize_file_content]-empty document: {file_path}")
        raise ToolException("The document content is empty.")
    return map_reduce_summarize(chunks, user_id, **kwargs)


def summary_file_content(file_path: str, user_id: str, **kwargs) -> str | None:
    if not os.path.exists(file_path):
        logger.warning(f"[summary_file_content]-document_path not exists: {file_path}")
        return "The file does not exist or the path is invalid."
    file_loader = DefaultFileLoaderComponent(file_path=file_path)
    doc = file_loader.load_full_content(include_metadata=True)
    page_content = doc.page_content
    direct_summary_limit = kwargs.get("direct_summary_token_threshold", DIRECT_SUMMARY_TOKEN_THRESHOLD)
    if count_tokens(page_content) <= direct_summary_limit:
        return direct_summary(page_content, user_id, **kwargs)
    else:
        return map_reduce_summarize_file_content(file_path, user_id, **kwargs)


class SummarizeKnowledgeBaseFileInput(BaseModel):
    file_sync_record_id: str = Field(
        description="File sync record ID. If not provided, call `get_build_task_files_detail_tool` first to obtain this parameter.",
        min_length=1,
        examples=["temp-a4a702df-24ad-42f8-bffb-6556f196",
                  "04d79ef39395476281e79ace89f3f584f71e1a33ed2a4e7eb8dda74301e0be51"],
    )
    knowledge_index_task_id: Optional[str] = Field(
        default=None,
        description="Optional parameter, knowledge index task ID associated with the file. If not provided, call `get_build_task_files_detail_tool` first to obtain this parameter.",
        examples=["6a7cefd5720241c2", "383c5210867a4931"],
    )


@tool(
    name_or_callable="summarize_knowledge_base_file_tool",
    description=(
            "Summarize the content of the specified file in knowledge base. "
            "**Important**: Before calling this tool, you MUST first call `get_build_task_files_detail_tool` "
            "to retrieve the file_sync_record_id, knowledge_index_task_id parameters. "
    ), args_schema=SummarizeKnowledgeBaseFileInput,
    response_format="content_and_artifact",
)
def summarize_knowledge_base_file_tool(
        file_sync_record_id: str,
        runtime: Annotated[ToolRuntime, InjectedToolArg],
        knowledge_index_task_id: Optional[str] = None,
) -> tuple[str, dict[str, str | int]]:
    user_id = runtime.context.user_id if runtime.context else "unknown"
    logger.info(
        f"[summarize_knowledge_base_file_tool] user_id={user_id}, file_sync_record_id={file_sync_record_id}, knowledge_index_task_id={knowledge_index_task_id}")
    file_info = get_build_task_file_detail(file_sync_record_id=file_sync_record_id,
                                           user_id=user_id,
                                           build_knowledge_index_task_id=knowledge_index_task_id)
    document_storage = None
    if knowledge_index_task_id and file_info.knowledge_index_folder_name:  # 查看是否存在已经存在的总结任务
        doc_store_path = os.path.join(KB_ROOT_PATH, file_info.knowledge_index_folder_name + DOC_STORE_SUFFIX)
        document_storage = GlobalDocumentContextStorage(root_path=doc_store_path)
        exists_results = document_storage.retriever_documents(ids=[knowledge_index_task_id])
        if exists_results and len(exists_results) > 0:
            return exists_results[0].page_content, {
                "summary_tokens": count_tokens(exists_results[0].page_content),
                "user_id": user_id,
            }

    file_path = file_info.file_path

    if not os.path.exists(file_path):
        error_msg = f"The file does not exist or the path is invalid: {file_path}"
        logger.warning(f"[summarize_knowledge_base_file_tool] {error_msg}")
        raise ToolException(error_msg)

    try:
        summary_result = summary_file_content(file_path=file_path, user_id=user_id,
                                              reduce_phase_template_without_metadata=True)
        if not summary_result:
            raise ToolException("Summarization failed, no result returned.")

        summary_tokens = count_tokens(summary_result)
        artifact = {
            "summary_tokens": summary_tokens,
            "user_id": user_id,
        }
        logger.info(
            f"[summarize_knowledge_base_file_tool] completed - tokens={summary_tokens}, user_id={user_id}"
        )
        if document_storage is not None and knowledge_index_task_id is not None:
            summary_doc = Document(
                id=knowledge_index_task_id,  # knowledge_index_task_id实际上就是文档的id
                page_content=summary_result
            )
            document_storage.save_documents([summary_doc])
            logger.info(
                f"[summarize_knowledge_base_file_tool]-Saved summary result to document storage, document_id: {knowledge_index_task_id}")
        return summary_result, artifact
    except ToolException:
        raise
    except Exception as exc:
        logger.exception(f"[summarize_knowledge_base_file_tool] unexpected error: {exc}")
        raise ToolException(f"Summarizing the file failed: {exc}")


@tool(
    name_or_callable="summarize_chat_session_uploaded_file_tool",
    description=(
            "Summarize the content of the specified file which uploaded by user in chat session. "
    ),
    response_format="content_and_artifact",
)
def summarize_chat_session_uploaded_file_tool(
        runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> tuple[str, dict[str, str | int]]:
    user_id = runtime.context.user_id if runtime.context else "unknown"
    chat_session_state = runtime.state.get("chat_session_state")
    if not chat_session_state or chat_session_state.uploaded_files_info is None:
        error_msg = "Required configuration is missing in the context when call the tool."
        logger.error(f"[summarize_chat_session_file_tool] {error_msg}")
        return "Failed to summarize chat session file: " + error_msg, {
            "summary_tokens": 0,
            "user_id": user_id,
        }

    uploaded_files_info: Dict[str, ChatSessionUploadFileInfo] = chat_session_state.uploaded_files_info

    # 如果没有上传任何文件
    if not uploaded_files_info or len(uploaded_files_info) == 0:
        return "No files have been uploaded in this chat session.", {
            "summary_tokens": 0,
            "user_id": user_id,
        }
    # 目前在聊天界面下只支持上传单个文件进行总结
    file_sync_record_id = list(uploaded_files_info.keys())[0]
    uploaded_file_info = uploaded_files_info[file_sync_record_id]
    file_name = uploaded_file_info.file_name
    chat_session_id = uploaded_file_info.chat_session_id
    logger.info(
        f"[summarize_chat_session_file_tool] user_id={user_id}, file_sync_record_id={file_sync_record_id}, file_name={file_name}")

    document_storage = None
    if file_sync_record_id:  # 查看是否存在已经存在的总结任务
        doc_store_path = os.path.join(KB_ROOT_PATH, CHAT_SESSION_TEMP_FILE_DIR_NAME, user_id, chat_session_id,
                                      "summary_doc_store")
        document_storage = GlobalDocumentContextStorage(root_path=doc_store_path)
        exists_results = document_storage.retriever_documents(ids=[file_sync_record_id])
        if exists_results and len(exists_results) > 0:
            return exists_results[0].page_content, {
                "summary_tokens": count_tokens(exists_results[0].page_content),
                "user_id": user_id,
                "file_name": file_name,
            }

    file_info = get_build_task_file_detail(file_sync_record_id=file_sync_record_id,
                                           user_id=user_id)
    file_path = file_info.file_path

    if not os.path.exists(file_path):
        error_msg = f"The file does not exist or the path is invalid: {file_path}"
        logger.warning(f"[summarize_chat_session_file_tool] {error_msg}")
        return "The file does not exist or the path is invalid.", {
            "summary_tokens": 0,
            "user_id": user_id,
            "file_name": file_name,
        }

    try:
        summary_result = summary_file_content(file_path=file_path, user_id=user_id,
                                              reduce_phase_template_without_metadata=True)
        if not summary_result:
            raise ToolException("Summarization failed, no result returned.")

        if document_storage is not None:
            summary_doc = Document(
                id=file_sync_record_id,  # file_sync_record_id实际上就是文档的id
                page_content=summary_result
            )
            document_storage.save_documents([summary_doc])
            logger.info(
                f"[summarize_chat_session_file_tool]-Saved summary result to document storage, document_id: {file_sync_record_id}")

        summary_tokens = count_tokens(summary_result)
        artifact = {
            "summary_tokens": summary_tokens,
            "user_id": user_id,
            "file_sync_record_id": file_sync_record_id,
            "file_name": file_name,
        }
        logger.info(
            f"[summarize_chat_session_file_tool] completed - tokens={summary_tokens}, user_id={user_id}"
        )
        return summary_result, artifact
    except ToolException:
        raise
    except Exception as exc:
        logger.exception(f"[summarize_chat_session_file_tool] unexpected error: {exc}")
        raise ToolException(f"Summarizing the file failed: {exc}")


def summary_text(text: str, user_id: str, **kwargs) -> str | None:
    direct_summary_limit = kwargs.get("direct_summary_token_threshold", DIRECT_SUMMARY_TOKEN_THRESHOLD)
    if count_tokens(text) <= direct_summary_limit:
        return direct_summary(text, user_id, **kwargs)
    else:
        chunk_size = kwargs.get("summary_chunk_size", SUMMARY_CHUNK_SIZE)
        chunk_overlap = kwargs.get("summary_chunk_overlap", SUMMARY_CHUNK_OVERLAP_TOKENS)
        is_chinese_text = kwargs.get("is_chinese_text", True)
        loader = ContentStringLoader(
            is_chinese_text=is_chinese_text,
            max_characters=chunk_size,
            overlap=chunk_overlap,
            content=text
        )
        docs = loader.load_and_split()
        text_chunks = [doc.page_content for doc in docs if doc.page_content]
        if not text_chunks:
            logger.warning("[summary_text]-empty text input.")
            return "The text is empty."
        return map_reduce_summarize(text_chunks, user_id, **kwargs)


class SummarizeTextInput(BaseModel):
    text: str = Field(
        description="The text content that needs to be summarized.",
        min_length=1,
        examples=["This is a long article about climate change...", "In this report, we analyze the market trends..."],
    )


@tool(
    name_or_callable="summarize_text",
    description="Summarize the provided text content. For long texts, Map-Reduce strategy will be used to generate structured summaries.",
    args_schema=SummarizeTextInput,
    response_format="content_and_artifact",
)
def summarize_text_tool(
        text: Annotated[str, Field(min_length=1)],
        runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> tuple[str, dict[str, str | int | float]]:
    user_id = runtime.context.user_id if runtime.context else "unknown"
    normalized_text = text.strip()
    logger.info(
        f"[summarize_text_tool] user_id={user_id}, text_length={len(normalized_text)}"
    )

    if not normalized_text:
        error_msg = "The provided text is empty."
        logger.warning(f"[summarize_text_tool] {error_msg}")
        raise ToolException(error_msg)

    try:
        original_tokens = count_tokens(normalized_text)
        summary_result = summary_text(text=normalized_text, user_id=user_id,
                                      reduce_phase_template_without_metadata=True)
        if not summary_result:
            raise ToolException("Summarization failed, no result returned.")

        summary_tokens = count_tokens(summary_result)
        artifact = {
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
            "compression_ratio": round(
                summary_tokens / original_tokens, 2
            ) if original_tokens else 0,
            "user_id": user_id,
        }
        logger.info(
            "[summarize_text_tool] completed - "
            f"original_tokens={original_tokens}, summary_tokens={summary_tokens}"
        )
        return summary_result, artifact
    except ToolException:
        raise
    except Exception as exc:
        logger.exception(f"[summarize_text_tool] unexpected error: {exc}")
        raise ToolException(f"Summarizing the text failed: {exc}")


def create_summarize_chat_session_uploaded_file_tool(chat_session_upload_file_info: ChatSessionUploadFileInfo):
    """
    创建一个自定义描述的聊天会话上传文件总结工具实例。

    此函数允许动态设置工具的描述信息,以适应不同的使用场景。

    Args:
        chat_session_upload_file_info: 聊天会话上传文件信息

    Returns:
        一个新的工具实例,具有自定义的描述

    Examples:
        >>> custom_tool = create_summarize_chat_session_uploaded_file_tool(
        ...     chat_session_upload_file_info=upload_info
        ... )
        >>> result = custom_tool.invoke({})
    """

    if chat_session_upload_file_info.is_from_extension:  # 来自浏览器扩展在页面刮取的网页内容
        uploaded_content = f"web content and the title is: {chat_session_upload_file_info.file_name}"

        metadata = chat_session_upload_file_info.file_metadata
        if metadata and isinstance(metadata, dict):
            uploaded_content += f", the web meta data is: {json.dumps(metadata, ensure_ascii=False)}"
    else:
        uploaded_content = f"file and the name is: {chat_session_upload_file_info.file_name}"

    description = (
        f"Summarize the content of the uploaded {uploaded_content}. "
        f"Use this tool to generate a comprehensive summary of the uploaded content."
    )

    try:
        result_map = ToolSchemaDefinition.load_builtin_tool_schema_dict_static(
            lang="en",
            tool_name="chat_session_summarize_tool",
            variables={"uploaded_content": uploaded_content}
        )

        description = result_map.get("description", description)
    except Exception as e:
        logger.exception(f"[create_summarize_chat_session_uploaded_file_tool] Load tool schema error: {e}")

    return summarize_chat_session_uploaded_file_tool.model_copy(update={"description": description})
