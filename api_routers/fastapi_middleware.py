from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from configs.model_config import logger, protected_paths, RUNNING_ENV
from .context import set_user_id


class UserContextMiddleware(BaseHTTPMiddleware):
    def _require_user_id_check_for_path(self, path: str) -> bool:
        return any(protected_path in path for protected_path in protected_paths)

    async def dispatch(self, request: Request, call_next):
        # 从请求头中获取用户ID
        user_id = request.headers.get("userid")

        # 检查是否是 local_knowledge_base_actions 路由
        if self._require_user_id_check_for_path(request.url.path):
            if RUNNING_ENV == "local":
                logger.info("Running in local environment, skipping user ID check for service path")
                user_id = "f681be09"
                set_user_id(user_id)
                return await call_next(request)
            if not user_id:
                logger.warning("User ID is missing in request headers for accessing service path")
                raise HTTPException(
                    status_code=401,
                    detail="User ID is missing in request headers for accessing service path"
                )

        # 将用户ID放入上下文
        set_user_id(user_id)
        # 处理请求
        response = await call_next(request)
        return response