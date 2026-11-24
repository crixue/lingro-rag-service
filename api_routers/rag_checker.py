import asyncio

from fastapi import APIRouter, Request

from api_routers.context import set_user_id
from configs.model_config import logger
from controllers.rag_checker_controller import RAGCheckerController
from controllers.vo.rag_checker_vo import RAGEvaluateRequestVO
from utils.response_util import server_response

router = APIRouter()


@router.post('/rag-checker/v1/evaluate')
async def rag_checker_evaluate(request_payload: RAGEvaluateRequestVO, request: Request, ):
    set_user_id(request_payload.user_id)
    logger.info(f"[rag_checker_evaluate]-request_payload: {request_payload}")
    controller = RAGCheckerController(request_payload.is_chinese_text,
                                      user_id=request_payload.user_id,
                                      product_id=request_payload.used_model_config.llm_id,
                                      rating_product_id=request_payload.rag_checker_config.rating_product_id
                                      )
    asyncio.create_task(controller.rag_evaluate_async(
        used_model_config=request_payload.used_model_config,
        retriever_config=request_payload.retriever_config,
        user_id=request_payload.user_id,
        **{
            **request_payload.rag_checker_config.model_dump(),
            "task_id": request_payload.task_id,
        }
    ))
    return server_response()
