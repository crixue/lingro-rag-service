import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from exceptions.custom_exception import NotHaveEnoughMoneyException
from configs.model_config import logger

def add_exception_handlers(app):
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        stack_trace = traceback.format_exc()
        logger.error(f'An error occurred: {exc}\n{stack_trace}')
        return JSONResponse(status_code=500, content={'code': -1, 'msg': str(exc), 'data': None})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={'code': -1, 'msg': str(exc), 'data': None})

    @app.exception_handler(NotHaveEnoughMoneyException)
    async def not_have_enough_money_exception_handler(request: Request, exc: NotHaveEnoughMoneyException):
        return JSONResponse(status_code=503, content={'code': 71001, 'msg': str(exc.message), 'data': None})