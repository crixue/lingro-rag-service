from fastapi import APIRouter, Query

from api_routers.context import set_user_id
from controllers.chat_session_controller import ChatSessionController
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.making_index_entrance_controller import BuildingIndexEntranceController
from controllers.vo.build_knowledge_index_request_vo import BuildKnowledgeIndexItemRequestVO, DeleteKnowledgeBaseIndexRequestVO
from utils.response_util import server_response

router = APIRouter()

@router.post('/knowledge-index/v1/build')
async def build_knowledge_index_async(request: BuildKnowledgeIndexItemRequestVO):
    set_user_id(request.user_id)
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(request.build_knowledge_index_type))
    if request.is_triggered_by_retry_task:
        await controller.retry_failed_task_async(request)
    else:
        await controller.building_index_async(request)

@router.delete('/knowledge-index/v1/drop')
async def del_some_indies_async(request: DeleteKnowledgeBaseIndexRequestVO):
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(request.build_knowledge_index_type))
    await controller.del_some_indies_async(request)

@router.delete('/knowledge-index/v1/drop-version-of-kg-index')
async def del_all_indies_and_docs_async(request: DeleteKnowledgeBaseIndexRequestVO):
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(request.build_knowledge_index_type))
    await controller.del_version_of_kg_index(request)
    return server_response()

@router.delete('/knowledge-index/v1/delete-workspace')
async def del_all_indies_and_docs_async(request: DeleteKnowledgeBaseIndexRequestVO):
    controller = BuildingIndexEntranceController(
        build_kg_index_type=BuildKnowledgeIndexTypeEnum(request.build_knowledge_index_type))
    await controller.del_workspace_async(request)
    return server_response()


@router.delete('/knowledge-index/v1/chat-session/drop')
def del_chat_session_indies_and_docs(chat_session_id: str = Query(..., description="chat session id"),
                                     user_id: str = Query(..., description="user id")):
    controller = ChatSessionController()
    controller.delete_chat_session_indies_and_docs(chat_session_id, user_id)
    return server_response()