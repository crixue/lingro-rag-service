import json
import os
from typing import Annotated, Optional, Any, Dict

from langchain.tools import tool, ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import ToolException, InjectedToolArg
from pydantic import BaseModel, Field

from api_routers.context import set_user_id
from configs.model_config import logger, DEFAULT_CONTEXT_TOKENS_LIMIT, \
    CHAT_SESSION_TEMP_FILE_BUILD_KNOWLEDGE_INDEX_CONFIG, CHAT_SESSION_TEMP_FILE_DIR_NAME, KB_ROOT_PATH
from controllers.common.common_constant import MetadataConstants, DOC_STORE_SUFFIX
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.making_index_entrance_controller import BuildingIndexEntranceController
from controllers.vo.build_knowledge_index_request_vo import BuildKnowledgeIndexItemRequestVO
from controllers.vo.startup_llm_model_request_vo import ChatSessionUploadFileInfo, RetrieverConfig
from core.common.common_constant import SEARCH_RELEVANT_DOCS_TOOL_NAME
from core.data_connections.retrievers.default_retriever_wrapper_v2 import DefaultRetrieverV2
from tools_definitions.lang import ToolSchemaDefinition
from tools_definitions.schemas.build_task_response import BuildTaskFileInfo
from tools_definitions.tools.tool_utils import get_build_task_file_detail
from utils.common_util import count_tokens


class RetrieverSearchInput(BaseModel):
    """Input schema for the retriever tool."""
    query: str = Field(
        description="Search query string. It should be a clear and specific question or keywords related to the information you want to find.",
        min_length=1,
        max_length=100,
        examples=["Python async programming", "什么是机器学习？", "2024年最新政策"],
    )


@tool(
    name_or_callable="search_relevant_documents_from_knowledge_base_tool",
    description=(
            "Search for relevant information from the knowledge base. "
            "Use this tool when you need to find information about current events or specific topics. "
    ),
    args_schema=RetrieverSearchInput,
    response_format="content_and_artifact",
)
def search_relevant_documents_from_knowledge_base_tool(
        query: Annotated[
            str,
            Field(
                description="Search query string. It should be a clear and specific question or keywords related to the information you want to find.",
                min_length=1,
                max_length=100,
            ),
        ],
        runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> tuple[str, dict[str, Any]]:
    """
    使用配置的检索器搜索相关文档。

    此工具会搜索知识库以查找与给定查询相关的文档。
    返回文档内容和关于搜索结果的元数据。

    Args:
        query: 搜索查询字符串
        runtime: 运行时配置（自动注入，包含 context 和 state）

    Returns:
        元组 (content_string, artifact_dict)，其中：
        - content_string: 格式化的文档内容
        - artifact_dict: 元数据，包括文档数量、tokens 等

    Raises:
        ToolException: 当检索器不可用或遇到错误时

    Examples:
        >>> result = search_relevant_documents_from_knowledge_base_tool.invoke({"query": "什么是 RAG？"})
        >>> content, artifact = result
        >>> print(f"找到 {artifact['retriever_docs_count']} 个相关文档")
    """
    user_id = runtime.context.user_id if runtime.context else "unknown"
    set_user_id(user_id)

    retriever_config = runtime.state.get("retriever_config")
    if not retriever_config:
        error_msg = "Retriever configuration is missing in the state."
        logger.error(f"[search_relevant_documents_tool] {error_msg}")
        raise ToolException(error_msg)

    return _execute_retrieval_and_format_results(
        query=query,
        runtime=runtime,
        retriever_config=retriever_config,
        user_id=user_id,
        tool_name="search_relevant_documents_from_knowledge_base_tool"
    )


def _format_document_for_llm(doc: Document) -> Optional[str]:
    """
    将文档内容和元数据包装成格式化的 JSON 字符串，供 LLM 使用。

    Args:
        doc: 要格式化的文档

    Returns:
        格式化的 JSON 字符串，如果文档无效则返回 None

    Examples:
        >>> from langchain_core.documents import Document
        >>> doc = Document(
        ...     page_content="示例内容",
        ...     metadata={"source": "example.pdf", "page": 1}
        ... )
        >>> result = _format_document_for_llm(doc)
        >>> print(result)
        {"content": "示例内容", "metadata": {"source": "example.pdf", "page": 1}}
    """
    if doc is None or doc.page_content is None:
        return None

    # 清理元数据（移除无用字段）
    cleaned_metadata = MetadataConstants.popup_useless_metadata(doc.metadata)

    # 构建文档信息
    feed_info = {
        "page_content": doc.page_content,
        "metadata": cleaned_metadata,
    }

    # 转换为 JSON 字符串
    return json.dumps(feed_info, ensure_ascii=False)


@tool(
    name_or_callable="search_relevant_documents_from_uploaded_files_tool",
    description=(
            "Search for relevant information from uploaded files/web contents by user. "
            "Use this tool when you need to find information about current events or specific topics. "
    ),
    args_schema=RetrieverSearchInput,
    response_format="content_and_artifact",
)
def search_relevant_documents_from_uploaded_files_tool(
        query: Annotated[
            str,
            Field(
                description="Search query string. It should be a clear and specific question or keywords related to the information you want to find in the uploaded files.",
                min_length=1,
                max_length=100,
            ),
        ],
        runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> tuple[str, dict[str, Any]]:
    """
    搜索用户上传文件中相关文档。

    此工具会搜索上传的文件以查找与给定查询相关的文档。
    返回文档内容和关于搜索结果的元数据。

    Args:
        query: 搜索查询字符串
        runtime: 运行时配置（自动注入，包含 context 和 state）

    Returns:
        元组 (content_string, artifact_dict)，其中：
        - content_string: 格式化的文档内容
        - artifact_dict: 元数据，包括文档数量、tokens 等
    Raises:
        ToolException: 当检索器不可用或遇到错误时

    Examples:
        >>> result = search_relevant_documents_from_uploaded_files_tool.invoke({"query": "文档中的关键结论是什么?"})
        >>> content, artifact = result
        >>> print(f"找到 {artifact['retriever_docs_count']} 个相关文档")
    """
    user_id = runtime.context.user_id if runtime.context else "unknown"
    set_user_id(user_id)

    chat_session_state = runtime.state.get("chat_session_state")
    if not chat_session_state or chat_session_state.uploaded_files_info is None:
        error_msg = "Required configuration is missing in the context when call the tool."
        logger.error(f"[search_relevant_documents_from_uploaded_files_tool] {error_msg}")
        return "Failed to summarize chat session file: " + error_msg, {
            "summary_tokens": 0,
            "user_id": user_id,
        }
    uploaded_files_info_dict: Dict[str, ChatSessionUploadFileInfo] = chat_session_state.uploaded_files_info

    if not uploaded_files_info_dict or len(uploaded_files_info_dict) == 0:
        return "No files have been uploaded in this chat session.", {
            "summary_tokens": 0,
            "user_id": user_id,
        }

    file_sync_record_id = list(uploaded_files_info_dict.keys())[0]
    uploaded_files_info = uploaded_files_info_dict.get(file_sync_record_id)
    chat_session_id = uploaded_files_info.chat_session_id

    parent_dir = os.path.join(CHAT_SESSION_TEMP_FILE_DIR_NAME, user_id, chat_session_id)
    knowledge_index_path = os.path.join(KB_ROOT_PATH, parent_dir)
    doc_store_path = os.path.join(KB_ROOT_PATH, parent_dir + DOC_STORE_SUFFIX)
    knowledge_index_type = BuildKnowledgeIndexTypeEnum.LOCAL_VECTOR_STORE

    retriever_config: RetrieverConfig = runtime.state.get("retriever_config")
    if not retriever_config:
        error_msg = "Retriever configuration is missing in the state."
        logger.error(f"[search_relevant_documents_tool] {error_msg}")
        raise ToolException(error_msg)

    retriever_config.retriever_type = knowledge_index_type
    retriever_config.knowledge_index_folder_id = chat_session_id
    retriever_config.knowledge_index_folder_name = parent_dir

    # 确保目录存在并且相应的目录下有文档存储,如果不存在则构建知识库索引
    if (not os.path.exists(knowledge_index_path) or not os.path.exists(doc_store_path) or
            len(os.listdir(doc_store_path)) == 0 or len(os.listdir(knowledge_index_path)) == 0):
        logger.info(
            f"[search_relevant_documents_from_uploaded_files_tool] Knowledge index path '{knowledge_index_path}' "
            f"or doc store path '{doc_store_path}' does not exist. Building knowledge index."
        )
        build_file_info: BuildTaskFileInfo = get_build_task_file_detail(file_sync_record_id=file_sync_record_id,
                                                                        user_id=user_id)
        request = BuildKnowledgeIndexItemRequestVO(
            id=chat_session_id,
            buildKnowledgeIndexConfig=CHAT_SESSION_TEMP_FILE_BUILD_KNOWLEDGE_INDEX_CONFIG,
            buildKnowledgeIndexType=knowledge_index_type,
            saveToPath=build_file_info.file_path,
            knowledgeIndexFolderId=chat_session_id,
            knowledgeIndexFolderName=parent_dir,
            triggerUserId=user_id,
            metadata=build_file_info.metadata,
            checkoutVersion=0,
            enableAnalyzeDocument=False,
        )
        controller = BuildingIndexEntranceController(
            build_kg_index_type=knowledge_index_type)
        controller.building_index(request)

    return _execute_retrieval_and_format_results(
        query=query,
        runtime=runtime,
        retriever_config=retriever_config,
        user_id=user_id,
        tool_name="search_relevant_documents_from_uploaded_files_tool"
    )


def create_search_relevant_documents_from_uploaded_files_tool(chat_session_upload_file_info: ChatSessionUploadFileInfo):
    """
    创建一个自定义描述的上传文件检索工具实例。

    此函数允许动态设置工具的描述信息，以适应不同的使用场景。

    Args:
        chat_session_upload_file_info

    Returns:
        一个新的工具实例，具有自定义的描述

    Examples:
        >>> custom_tool = create_search_relevant_documents_from_uploaded_files_tool(
        ...     description="Search for technical information in uploaded PDF files."
        ... )
        >>> result = custom_tool.invoke({"query": "API documentation"})
    """

    if chat_session_upload_file_info.is_from_extension:  # 来自浏览器扩展在页面刮取的网页内容
        uploaded_content = f"web content and the title is: {chat_session_upload_file_info.file_name}"

        metadata = chat_session_upload_file_info.file_metadata
        if metadata and isinstance(metadata, dict):
            uploaded_content += f", the web meta data is: {json.dumps(metadata, ensure_ascii=False)}"
    else:
        uploaded_content = f"file and the name is: {chat_session_upload_file_info.file_name}"

    description = (
        f"Search for relevant information from the uploaded files/web contents by user, "
        f"Use this tool when you need to find information about current events or specific topics. "
    )
    try:
        result_map = ToolSchemaDefinition.load_builtin_tool_schema_dict_static(
            lang="en",
            tool_name="chat_session_retriever_tool",
            variables={"uploaded_content": uploaded_content}
        )

        description = result_map.get("description", "")
    except Exception as e:
        description = (
            f"Search for relevant information from the uploaded files/web contents by user, "
            f"Use this tool when you need to find information about current events or specific topics. "
        )
        logger.exception(f"[create_search_relevant_documents_from_uploaded_files_tool] Load tool schema error: {e}")

    return search_relevant_documents_from_uploaded_files_tool.model_copy(update={"description": description})


def _execute_retrieval_and_format_results(
        query: str,
        runtime: ToolRuntime,
        retriever_config: RetrieverConfig,
        user_id: str,
        tool_name: str
) -> tuple[str, dict[str, Any]]:
    """
    执行检索并格式化结果的通用函数。

    Args:
        query: 搜索查询字符串
        runtime: 运行时配置
        retriever_config: 检索器配置
        user_id: 用户ID
        tool_name: 工具名称（用于日志）

    Returns:
        元组 (content_string, artifact_dict)

    Raises:
        ToolException: 当检索器不可用或遇到错误时
    """
    # 获取 LLM 客户端
    llm_client = runtime.context.llm_client if runtime.context else None
    if not llm_client:
        error_msg = "LLM client is missing in the context."
        logger.error(f"[{tool_name}] {error_msg}")
        raise ToolException(error_msg)

    # 获取上下文 token 限制
    context_tokens_limit = runtime.state.get("context_tokens_limit", DEFAULT_CONTEXT_TOKENS_LIMIT)

    retriever_docs_info: dict[str, Document] = {}
    keywords: list[str] = [query]

    logger.info(
        f"[{tool_name}] Start searching for query '{query}' "
        f"for user_id '{user_id}' using config: {retriever_config.model_dump()}"
    )

    # 执行检索
    try:
        custom_retriever = DefaultRetrieverV2(
            llm=llm_client,
            query_str=query,
            **retriever_config.model_dump(),
        )
        docs = custom_retriever.get_relevant_documents(query)
    except Exception as e:
        logger.exception(f"[{tool_name}] Retriever error: {e}")
        raise ToolException(
            f"The retriever tool '{SEARCH_RELEVANT_DOCS_TOOL_NAME}' is currently unavailable. Please try again later. Error: {str(e)}"
        )

    # 处理和格式化文档
    formatted_docs = ""
    total_tokens = 0

    if docs:
        for index, doc in enumerate(docs):
            doc_info = _format_document_for_llm(doc)
            if doc_info is None:
                continue

            doc_tokens = count_tokens(doc_info)
            total_tokens += doc_tokens

            if total_tokens > context_tokens_limit:
                logger.info(
                    f"[{tool_name}] Query '{query}' exceeded token limit "
                    f"{context_tokens_limit}. Current total tokens: {total_tokens}. "
                    f"Processed {index} documents."
                )
                break

            formatted_docs += doc_info + "\n"

            if doc.id is not None:
                retriever_docs_info[doc.id] = doc

    # 准备搜索关键词
    search_keywords = "".join(keywords)

    # 构建 artifact（元数据）
    artifact = {
        "retriever_docs_info": list(retriever_docs_info.values()),
        "search_keywords": search_keywords,
        "retriever_docs_count": len(retriever_docs_info),
        "retriever_docs_total_tokens": total_tokens,
    }

    logger.info(
        f"[{tool_name}] Search completed - "
        f"Found {len(retriever_docs_info)} documents, "
        f"Total tokens: {total_tokens}"
    )

    return formatted_docs, artifact
