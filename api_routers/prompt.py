from fastapi import APIRouter

from controllers.serialize_controller import DefaultSerializeController
from controllers.vo.build_knowledge_index_request_vo import SerializePromptRequestVO
from utils.response_util import server_response

router = APIRouter()

@router.put('/prompt/v1/serialize')
def serialize_prompt(request_payload: SerializePromptRequestVO):
    prompt_info = DefaultSerializeController(prompt_folder_name=request_payload.prompt_folder_name).serialize(**request_payload.serialize_info)
    return server_response(data=prompt_info)