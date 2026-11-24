from typing import Optional

import requests
from langchain_core.tools import ToolException
from pydantic import ValidationError

from configs.model_config import JAVA_MICRO_SERVICE_URL, logger
from tools_definitions.schemas.build_task_response import BuildTaskFileInfo


def get_build_task_file_detail(
        file_sync_record_id: str,
        user_id: str,
        build_knowledge_index_task_id: Optional[str] = None,
        timeout: int = 30,
) -> BuildTaskFileInfo:
    """
    调用微服务接口获取单个构建任务的文件信息。
    """
    if not file_sync_record_id:
        error_msg = "file_sync_record_id is required parameter"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    base_url = JAVA_MICRO_SERVICE_URL.get("get-build-task-file-detail")
    if not base_url:
        error_msg = "Missing microservice URL configuration for 'get-build-task-file-detail'"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    params = {
        "buildKnowledgeIndexTaskId": build_knowledge_index_task_id,
        "fileSyncRecordId": file_sync_record_id,
    }
    headers = {
        "accept": "*/*",
        "userid": user_id,
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        error_msg = f"HTTP response failed: {exc}"
        logger.exception(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    try:
        payload = response.json()
    except ValueError as exc:
        error_msg = f"Parsing response JSON failed: {exc}"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    if payload.get("code") != 0:
        error_msg = f"API returned error code {payload.get('code')}: {payload.get('msg')}"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    data = payload.get("data")
    if not data:
        error_msg = "API response data is empty"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    try:
        file_info = BuildTaskFileInfo(**data)
    except ValidationError as exc:
        error_msg = f"Validating response data failed: {exc}"
        logger.error(f"[get_build_task_file_detail] {error_msg}")
        raise ToolException(error_msg)

    logger.info(
        "[get_build_task_file_detail] succeed "
        f"task_id={file_info.knowledge_index_task_id}, file_sync_record_id={file_info.file_sync_record_id}"
    )
    return file_info
