import asyncio
import json
import os

import requests

from configs.model_config import BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG, JAVA_MICRO_SERVICE_URL, logger
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.enums.make_knowledge_base_enum import BuildKnowledgeIndexStatusEnum
from controllers.making_index_entrance_controller import BuildingIndexEntranceController
from controllers.vo.build_knowledge_index_request_vo import BuildKnowledgeIndexBundleRequestVO
from utils.redis_client import DefaultRedisClient

ENABLE_INDEX_SCHEDULER = os.getenv("ENABLE_INDEX_SCHEDULER", "true").lower() in {"1", "true", "yes"}


def _get_redis_client_or_none():
    """索引任务调度依赖 Redis 队列。Redis 不可用时降级为空转，避免后台任务持续抛异常。"""
    if not ENABLE_INDEX_SCHEDULER:
        logger.info("[index-scheduler]-ENABLE_INDEX_SCHEDULER=false, index task scheduler is disabled.")
        return None
    try:
        redis_client = DefaultRedisClient().get_redis_client()
        redis_client.ping()
        return redis_client
    except Exception as e:
        logger.warning(f"[index-scheduler]-Redis is unavailable, index task scheduler is disabled: {e}")
        return None


async def building_bundle_of_index_task_scheduler():
    redis_client = _get_redis_client_or_none()
    if redis_client is None:
        return
    consume_task_count = BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_count"]
    while consume_task_count > 0:
        for _ in range(consume_task_count):
            try:
                task_request_json = redis_client.rpop \
                    (BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_build_kg_index_queue_name"])
            except Exception as e:
                logger.warning(f"[building_bundle_of_index_task_scheduler]-Read queue failed, retrying: {e}")
                await asyncio.sleep(BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_interval_seconds"])
                break
            if task_request_json is None:
                logger.debug("[building_bundle_of_index_task_scheduler]-No tasks found in queue.")
                await asyncio.sleep(BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_interval_seconds"])
                break
            else:
                request: BuildKnowledgeIndexBundleRequestVO = BuildKnowledgeIndexBundleRequestVO.model_validate_json \
                    (task_request_json)
                result = await building_bundle_of_index(request)
                try:
                    requests.put(url=JAVA_MICRO_SERVICE_URL["notice-version-checkout"],
                                 headers={"Content-Type": "application/json"},
                                 json=result)
                    logger.info(
                        f"[building_bundle_of_index_task_scheduler]-Notice core-service the knowledge_index_folder_id-{request.knowledge_index_folder_id} "
                        f"mvc version-{request.knowledge_index_version} have build complete")
                except Exception as e:
                    logger.exception(
                        f"Failed to notice version of KG build complete for folder id: {request.knowledge_index_folder_id} with ex: {e}")


async def building_bundle_of_index(request: BuildKnowledgeIndexBundleRequestVO) -> dict:
    request_list = request.build_knowledge_index_item_request_list
    results = []
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(request.build_knowledge_index_type))
    strategy = controller.building_strategy
    for req in request_list:
        logger.info \
            (f"[building_bundle_of_index]-Start to build bundle of index for id:{req.id} folder id: {req.knowledge_index_folder_id}")
        result = await strategy.execute_with_preprocessing_async(req)
        results.append(result)
    success_build_count = len([result for result in results if result.build_status == BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value])
    failed_build_count = len([result for result in results if result.build_status != BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value])
    resp: dict = {
        "successBuildCount": success_build_count,
        "failedBuildCount": failed_build_count,
        "knowledgeIndexFolderId": request.knowledge_index_folder_id,
        "knowledgeIndexVersion": request.knowledge_index_version,
        "knowledgeIndexMVControlId": request.knowledge_index_mv_control_id
    }
    require_to_delete_requests = request.require_to_delete_version_requests
    if require_to_delete_requests and len(require_to_delete_requests) > 0:
        for delete_request in require_to_delete_requests:
            strategy.del_version_of_indies_and_docs(delete_request)
            logger.info(f"[building_bundle_of_index]-Delete version-{delete_request.checkout_version} of index for folder id: {delete_request.knowledge_index_folder_id}")
    return resp

async def building_stand_alone_index_task_scheduler():
    redis_client = _get_redis_client_or_none()
    if redis_client is None:
        return
    consume_task_count = BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_count"]
    while consume_task_count > 0:
        for _ in range(consume_task_count):
            try:
                task_request_json = redis_client.rpop \
                    (BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_standalone_build_kg_index_queue_name"])
                if task_request_json is None:
                    logger.debug("[building_stand_alone_index_task_scheduler]-No tasks found in queue.")
                    await asyncio.sleep(BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_interval_seconds"])
                    break
                else:
                    request: BuildKnowledgeIndexBundleRequestVO = BuildKnowledgeIndexBundleRequestVO.model_validate_json \
                        (task_request_json)
                    await building_stand_alone_index(request)
                    logger.info(f"[building_stand_alone_index_task_scheduler]-Successfully processed standalone index for folder: {request.knowledge_index_folder_id}")
            except Exception as e:
                logger.exception(f"[building_stand_alone_index_task_scheduler]-Error processing task: {e}")
                await asyncio.sleep(BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG["consume_task_interval_seconds"])

async def building_stand_alone_index(request: BuildKnowledgeIndexBundleRequestVO):
    request_list = request.build_knowledge_index_item_request_list
    if len(request_list) == 0:
        logger.info("[building_stand_alone_index]-No tasks found in request list.")
        return
    results = []
    item = request_list[0]
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(item.build_knowledge_index_type))
    for req in request_list:
        logger.info(
            f"[building_stand_alone_index]-Start to build standalone of index for id:{req.id} folder name: {req.knowledge_index_folder_name} "
            f"and is_triggered_by_retry_task: {req.is_triggered_by_retry_task}")
        if req.is_triggered_by_retry_task:
            result = controller.retry_failed_task(req)
        else:
            result = await controller.building_strategy.execute_with_preprocessing_async(req)
        results.append(result)