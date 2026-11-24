import json
import os
from typing import Annotated, Any, Optional, Dict

from langchain.tools import tool, ToolRuntime
from langchain_core.tools import ToolException, InjectedToolArg
from pydantic import BaseModel, Field

from api_routers.context import set_user_id
from configs.model_config import logger
from controllers.vo.startup_llm_model_request_vo import ChatSessionUploadFileInfo, ChatSessionState
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from tools_definitions.tools.tool_utils import get_build_task_file_detail


class ReadFileContentInput(BaseModel):
    """Input schema for the read file content tool."""
    file_sync_record_id: str = Field(
        description="The file sync record ID to identify which file to read.",
        min_length=1,
    )
    page: int = Field(
        default=1,
        description="The page number to read (starts from 1).",
        ge=1,
    )
    max_tokens: int = Field(
        default=50000,
        description="Maximum number of tokens to read per page.",
        ge=100,
        le=100000,
    )


class ReadUploadedFileContentInput(BaseModel):
    """Input schema for reading uploaded file content."""
    page: int = Field(
        default=1,
        description="The page number to read (starts from 1).",
        ge=1,
    )
    max_tokens: int = Field(
        default=50000,
        description="Maximum number of tokens to read per page.",
        ge=100,
        le=100000,
    )


def _execute_file_reading(
        file_sync_record_id: str,
        page: int,
        max_tokens: int,
        user_id: str,
        tool_name: str
) -> tuple[str, dict[str, Any]]:
    """
    执行文件读取的通用函数。

    Args:
        file_sync_record_id: 文件同步记录 ID
        page: 页码（从 1 开始）
        max_tokens: 每页最大 token 数量
        user_id: 用户 ID
        tool_name: 工具名称（用于日志）

    Returns:
        元组 (content_string, artifact_dict)

    Raises:
        ToolException: 当文件不存在或读取失败时
    """
    logger.info(
        f"[{tool_name}] Reading file for sync_record_id='{file_sync_record_id}', "
        f"page={page}, max_tokens={max_tokens}, user_id='{user_id}'"
    )

    try:
        # 获取文件信息
        build_file_info = get_build_task_file_detail(
            file_sync_record_id=file_sync_record_id,
            user_id=user_id
        )

        if not build_file_info or not build_file_info.file_path:
            error_msg = f"File not found for sync_record_id: {file_sync_record_id}"
            logger.error(f"[{tool_name}] {error_msg}")
            raise ToolException(error_msg)

        file_path = build_file_info.file_path

        # 检查文件是否存在
        if not os.path.exists(file_path):
            error_msg = f"File path does not exist: {file_path}"
            logger.error(f"[{tool_name}] {error_msg}")
            raise ToolException(error_msg)

        # 使用 DefaultFileLoaderComponent 加载文件内容
        loader = DefaultFileLoaderComponent(file_path=file_path)
        document, pagination_info = loader.load_tokens_of_content(
            max_tokens=max_tokens,
            page=page,
            include_metadata=True
        )

        # 构建返回内容
        content = document.page_content

        # 构建 artifact（元数据）
        artifact = {
            "file_name": build_file_info.file_name,
            "file_path": file_path,
            "file_sync_record_id": file_sync_record_id,
            "current_page": pagination_info["current_page"],
            "page_tokens": pagination_info["page_tokens"],
            "total_tokens": pagination_info["total_tokens"],
            "has_next_page": pagination_info["has_next_page"],
            "max_tokens_per_page": pagination_info["max_tokens_per_page"],
            "pagination_mode": pagination_info.get("pagination_mode", "element_based"),
        }

        logger.info(
            f"[{tool_name}] Successfully read file - "
            f"Page {page}, Tokens: {pagination_info['page_tokens']}/{pagination_info['total_tokens']}, "
            f"Has next page: {pagination_info['has_next_page']}, "
            f"Mode: {pagination_info.get('pagination_mode', 'element_based')}"
        )

        # 构建描述性返回信息
        summary = (
            f"File: {build_file_info.file_name}\n"
            f"Current page: {page}\n"
            f"Page tokens: {pagination_info['page_tokens']}\n"
            f"Total tokens: {pagination_info['total_tokens']}\n"
            f"Has next page: {'Yes' if pagination_info['has_next_page'] else 'No'}\n"
            f"Pagination mode: {pagination_info.get('pagination_mode', 'element_based')}\n\n"
            f"Content:\n{content}"
        )

        return summary, artifact

    except ToolException:
        raise
    except Exception as e:
        error_msg = f"Failed to read file content: {str(e)}"
        logger.exception(f"[{tool_name}] {error_msg}")
        raise ToolException(error_msg)


@tool(
    name_or_callable="read_file_content_from_knowledge_tool",
    description=(
            "Read file content from the knowledge base. "
            "This tool reads file content in pages based on token limits. "
            "Use this when you need to read large files incrementally. "
            "The tool will return the current page content along with pagination information "
            "including current page tokens, total tokens, and whether there is a next page."
    ),
    args_schema=ReadFileContentInput,
    response_format="content_and_artifact",
)
def read_file_content_from_knowledge_tool(
        runtime: Annotated[ToolRuntime, InjectedToolArg],
        file_sync_record_id: Annotated[
            str,
            Field(
                description="The file sync record ID to identify which file to read.",
                min_length=1,
            ),
        ],
        page: Annotated[
            int,
            Field(
                default=1,
                description="The page number to read (starts from 1).",
                ge=1,
            ),
        ] = 1,
        max_tokens: Annotated[
            int,
            Field(
                default=50000,
                description="Maximum number of tokens to read per page.",
                ge=100,
                le=100000,
            ),
        ] = 50000,
) -> tuple[str, dict[str, Any]]:
    """
    按页读取知识库文件内容。

    此工具根据 token 限制分页读取知识库中的文件内容。
    返回当前页的内容和包含分页信息的元数据。

    Args:
        file_sync_record_id: 文件同步记录 ID
        page: 页码（从 1 开始）
        max_tokens: 每页最大 token 数量
        runtime: 运行时配置（自动注入）

    Returns:
        元组 (content_string, artifact_dict)，其中：
        - content_string: 当前页的文件内容
        - artifact_dict: 分页元数据，包括当前页 tokens、总 tokens、是否有下一页等

    Raises:
        ToolException: 当文件不存在或读取失败时

    Examples:
        >>> result = read_file_content_from_knowledge_tool.invoke({
        ...     "file_sync_record_id": "abc123",
        ...     "page": 1,
        ...     "max_tokens": 4000
        ... })
        >>> content, artifact = result
        >>> print(f"当前页 tokens: {artifact['page_tokens']}")
        >>> print(f"是否有下一页: {artifact['has_next_page']}")
    """
    user_id = runtime.context.user_id if runtime.context else "unknown"
    set_user_id(user_id)

    return _execute_file_reading(
        file_sync_record_id=file_sync_record_id,
        page=page,
        max_tokens=max_tokens,
        user_id=user_id,
        tool_name="read_file_content_from_knowledge_tool"
    )


@tool(
    name_or_callable="read_uploaded_file_content_tool",
    description=(
            "Read content from the files/web pages uploaded by the user in this chat session. "
            "This tool reads file content in pages based on token limits. "
            "Use this when you need to read the uploaded file content incrementally. "
            "The tool will automatically identify the uploaded file in the current chat session. "
            "Returns the current page content along with pagination information "
            "including current page tokens, total tokens, and whether there is a next page."
    ),
    args_schema=ReadUploadedFileContentInput,
    response_format="content_and_artifact",
)
def read_uploaded_file_content_tool(
        runtime: Annotated[ToolRuntime, InjectedToolArg],
        page: Annotated[
            int,
            Field(
                default=1,
                description="The page number to read (starts from 1).",
                ge=1,
            ),
        ] = 1,
        max_tokens: Annotated[
            int,
            Field(
                default=50000,
                description="Maximum number of tokens to read per page.",
                ge=100,
                le=100000,
            ),
        ] = 50000,
) -> tuple[str, dict[str, Any]]:
    """
    按页读取用户上传的文件内容。

    此工具从当前聊天会话中获取用户上传的文件信息，并根据 token 限制分页读取文件内容。
    返回当前页的内容和包含分页信息的元数据。

    Args:
        page: 页码（从 1 开始）
        max_tokens: 每页最大 token 数量
        runtime: 运行时配置（自动注入）

    Returns:
        元组 (content_string, artifact_dict)，其中：
        - content_string: 当前页的文件内容
        - artifact_dict: 分页元数据，包括当前页 tokens、总 tokens、是否有下一页等

    Raises:
        ToolException: 当文件不存在或读取失败时

    Examples:
        >>> result = read_uploaded_file_content_tool.invoke({
        ...     "page": 1,
        ...     "max_tokens": 4000
        ... })
        >>> content, artifact = result
        >>> print(f"当前页 tokens: {artifact['page_tokens']}")
        >>> print(f"是否有下一页: {artifact['has_next_page']}")
    """
    user_id = runtime.context.user_id if runtime.context else "unknown"
    set_user_id(user_id)

    # 从 state 中获取聊天会话状态
    chat_session_state: Optional[ChatSessionState] = runtime.state.get("chat_session_state")
    if not chat_session_state or chat_session_state.uploaded_files_info is None:
        error_msg = "No files have been uploaded in this chat session."
        logger.error(f"[read_uploaded_file_content_tool] {error_msg}")
        raise ToolException(error_msg)

    uploaded_files_info_dict: Dict[str, ChatSessionUploadFileInfo] = chat_session_state.uploaded_files_info

    if not uploaded_files_info_dict or len(uploaded_files_info_dict) == 0:
        error_msg = "No files found in uploaded files info."
        logger.error(f"[read_uploaded_file_content_tool] {error_msg}")
        raise ToolException(error_msg)

    # 获取第一个上传的文件信息
    file_sync_record_id = list(uploaded_files_info_dict.keys())[0]
    uploaded_file_info = uploaded_files_info_dict.get(file_sync_record_id)

    logger.info(
        f"[read_uploaded_file_content_tool] Reading uploaded file: "
        f"file_name='{uploaded_file_info.file_name}', "
        f"file_sync_record_id='{file_sync_record_id}', "
        f"is_from_extension={uploaded_file_info.is_from_extension}"
    )

    return _execute_file_reading(
        file_sync_record_id=file_sync_record_id,
        page=page,
        max_tokens=max_tokens,
        user_id=user_id,
        tool_name="read_uploaded_file_content_tool"
    )
