from typing import Annotated
import requests
import json
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import ToolException, InjectedToolArg
from pydantic import BaseModel, Field, ValidationError

from configs.model_config import logger, JAVA_MICRO_SERVICE_URL
from tools_definitions.schemas.build_task_response import (
    BuildTaskFilesResponse,
)

def can_use_get_build_task_files_detail_tool(user_id: str, knowledge_index_folder_id: str, checkout_version: str) -> bool:
    return all([user_id, knowledge_index_folder_id, checkout_version])


class GetBuildTaskFileDetailInput(BaseModel):
    page_num: Annotated[int, Field(
        description="Page number, starting from 1",
        ge=1,
        default=1,
        examples=[1],
    )] = 1


@tool(
    name_or_callable="get_knowledge_base_files_detail",
    description="Retrieve a paginated list of file information for the current knowledge base build tasks. Returns task ID(knowledge_index_task_id), file sync record ID(file_sync_record_id), file name(file_name), creation time(created_at), and other details.",
    args_schema=GetBuildTaskFileDetailInput,
    response_format="content",
)
def get_knowledge_base_files_detail_tool(
    page_num: Annotated[int, Field(ge=1)] = 1,
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> str:
    """
    分页获取知识库构建任务的文件信息列表

    Args:
        knowledge_index_folder_id: 知识库索引文件夹ID
        checkout_version: 检出版本号
        page_num: 页码，默认1
        runtime: 运行时上下文

    Returns:
        formatted_result: 格式化的结果字符串
    """
    if not runtime or not runtime.context:
        logger.warning("[get_knowledge_base_files_detail_tool] runtime or runtime.context is None")
        raise ToolException("Tool execution context is missing")
    context = runtime.context
    user_id = context.user_id
    retrieve_config_state = runtime.state.get("retriever_config")
    if not retrieve_config_state:
        error_msg = "Retriever configuration is missing in the context"
        logger.error(f"[get_knowledge_base_files_detail_tool] {error_msg}")
        return "Failed to get knowledge base files detail: " + error_msg

    knowledge_index_folder_id = retrieve_config_state.knowledge_index_folder_id
    checkout_version = retrieve_config_state.version

    if knowledge_index_folder_id is None or checkout_version is None:
        error_msg = "Required parameter is missing in the context"
        logger.error(f"[get_knowledge_base_files_detail_tool] {error_msg}")
        raise ToolException(error_msg)

    logger.info(
        f"[get_knowledge_base_files_detail_tool] user_id={user_id}, "
        f"knowledge_index_folder_id={knowledge_index_folder_id}, "
        f"checkout_version={checkout_version}, page_num={page_num}"
    )

    try:
        # Build request URL
        base_url = JAVA_MICRO_SERVICE_URL["get-build-task-files-detail"]
        params = {
            "knowledgeIndexFolderId": knowledge_index_folder_id,
            "checkoutVersion": checkout_version,
            "pageNum": page_num,
        }
        headers = {
            "accept": "*/*",
            "userid": user_id,
        }

        # Send HTTP request
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse response using Pydantic model
        try:
            response_data = BuildTaskFilesResponse(**response.json())
        except ValidationError as e:
            error_msg = f"Response data validation failed: {str(e)}"
            logger.error(f"[get_knowledge_base_files_detail_tool] {error_msg}")
            raise ToolException(error_msg)

        if response_data.code != 0:
            error_msg = f"API response failed: {response_data.msg}"
            logger.error(f"[get_knowledge_base_files_detail_tool] {error_msg}")
            raise ToolException(error_msg)

        data = response_data.data
        if not data or not data.list:
            error_msg = "API returned empty data field"
            logger.error(f"[get_knowledge_base_files_detail_tool] {error_msg}")
            raise ToolException(error_msg)

        # Convert to dict with descriptions and serialize to JSON
        result_dict = data.to_dict_with_descriptions()
        def _remove_none(obj):
            if isinstance(obj, dict):
                return {k: _remove_none(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [_remove_none(v) for v in obj if v is not None]
            return obj

        cleaned = _remove_none(result_dict)
        formatted_result = json.dumps(cleaned, ensure_ascii=False)

        logger.info(
            f"[get_knowledge_base_files_detail_tool] completed - "
            f"total={data.total}, file_count={len(data.list)}, user_id={user_id} formatted_result={formatted_result}"
        )

        return formatted_result

    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP request failed: {str(e)}"
        logger.exception(f"[get_knowledge_base_files_detail_tool] {error_msg}")
        raise ToolException(error_msg)
    except ToolException:
        raise
    except Exception as e:
        error_msg = f"Failed to retrieve file information: {str(e)}"
        logger.exception(f"[get_knowledge_base_files_detail_tool] {error_msg}")
        raise ToolException(error_msg)
