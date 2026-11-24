from contextvars import ContextVar
from typing import Optional

# 创建上下文变量存储用户ID
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

def get_user_id() -> Optional[str]:
    """获取当前请求的用户ID"""
    return user_id_var.get()

def set_user_id(user_id: Optional[str]) -> None:
    """设置当前请求的用户ID"""
    user_id_var.set(user_id)
