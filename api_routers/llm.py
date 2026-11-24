import glob
import os

import requests
from fastapi import APIRouter, Request, HTTPException, Depends
from starlette.responses import StreamingResponse

from configs.model_config import logger, JAVA_MICRO_SERVICE_URL
from controllers.vo.startup_llm_model_request_vo import ChatRequestVO
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from service.chat_agent_service import DefaultChatAgentService
from utils.response_util import server_response

router = APIRouter()

def pre_request_check(request: Request):
    user_id = request.headers.get("userid", None)
    if not user_id:
        logger.warning("User ID is missing in request headers for accessing service path")
        raise HTTPException(
            status_code=401,
            detail="User ID is missing in request headers for accessing service path"
        )
    product_id = request.headers.get("productId", None)
    resp = requests.get(
        url=JAVA_MICRO_SERVICE_URL["check-user-has-enough-money"] + f"/{user_id}?productId={product_id}",
        headers={"Content-Type": "application/json"},
    )
    if resp.status_code != 200 or resp.json().get("code", -1) != 0:
        err_msg = resp.json().get("msg", -1)
        raise Exception(f"Error occurred do pre check: {err_msg}")
    result = resp.json()["data"]
    if not result:
        raise HTTPException(
            status_code=402,
            detail="Not have enough money, can not use this service."
        )


@router.post('/llm-backend/v1/chat-stream/completions')
async def completions_stream(request_payload: ChatRequestVO,
                             request: Request,
                             _: bool = Depends(pre_request_check)):
    user_id = request.headers.get("userid", None)
    init_model_request, model_info = find_model_info(request_payload)
    langchain_llm = DefaultLangchainLLM(model_name=model_info.get("name"), user_id=user_id, product_id=model_info.get("id"),
                                     model_config=init_model_request.llm_model_config)
    agent_service = DefaultChatAgentService(langchain_llm=langchain_llm)
    response = StreamingResponse(content=agent_service.completions_stream(chat_request=request_payload, ),
                                 media_type="text/event-stream; charset=utf-8")
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@router.post('/llm-backend/v1/mock-chat-stream/completions')
async def mock_completions_stream(request: Request, _: bool = Depends(pre_request_check)):
    def mock_stream():
        mock_files = glob.glob('mock_data/*')
        import random
        random_file = mock_files[random.randint(0, len(mock_files) - 1)]
        with open(random_file, 'r', encoding='utf-8') as file:
            for line in file:
                import time
                time.sleep(0.01)
                yield line

    response = StreamingResponse(content=mock_stream(), media_type="text/event-stream; charset=utf-8")
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@router.post('/llm-backend/v1/chat/completions')
async def completions(request_payload: ChatRequestVO, request: Request):
    user_id = request.headers.get("userid", None)
    init_model_request, model_info = find_model_info(request_payload)
    langchain_llm = DefaultLangchainLLM(model_name=model_info.get("name"), user_id=user_id, product_id=model_info.get("id"), model_config=init_model_request.llm_model_config)
    agent_service = DefaultChatAgentService(langchain_llm=langchain_llm)
    resp = await agent_service.completions(chat_request=request_payload)
    logger.debug(f"completions response: {resp.dict()}")
    return server_response(data=resp.dict())

def find_model_info(request_payload):
    init_model_request = request_payload.used_model_config
    model_info = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=init_model_request.llm_id)
    if init_model_request.llm_id is None:
        init_model_request.llm_id = model_info.get("id")
    init_model_request.llm_model_name = model_info.get("name")
    return init_model_request, model_info