import asyncio
import pymysql
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
import nltk

from api_routers.fastapi_middleware import UserContextMiddleware

# 先导入 model_config 以触发环境变量验证
try:
    from configs.model_config import logger, NLTK_DATA_PATH, MYSQL_CONFIG, REDIS_CONFIG
except EnvironmentError as e:
    print(f"❌ App Start Failed caused by environment error: {e}")
    exit(1)

from controllers.schedulers.build_knowledge_index_schedule import building_bundle_of_index_task_scheduler, \
    building_stand_alone_index_task_scheduler
from exceptions.fastapi_exception_handlers import add_exception_handlers
from api_routers import llm, plugins, local_knowledge_base_actions, prompt, rag_checker, mcp

# 使用配置的 NLTK 数据路径
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
nltk.data.path = [NLTK_DATA_PATH]

# 服务就绪标志，lifespan 启动完成后设为 True
_service_ready = False
_thread_pool = ThreadPoolExecutor(max_workers=4)


def _check_mysql() -> dict:
    """同步检查 MySQL 连接（使用 PyMySQL）"""
    try:
        conn = pymysql.connect(
            host=MYSQL_CONFIG["MYSQL_HOST"],
            port=MYSQL_CONFIG["MYSQL_PORT"],
            user=MYSQL_CONFIG["MYSQL_USER"],
            password=MYSQL_CONFIG["MYSQL_PASSWORD"],
            database=MYSQL_CONFIG["MYSQL_DBNAME"],
            charset=MYSQL_CONFIG["MYSQL_CHARSET"],
            connect_timeout=3,
        )
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"Health check - MySQL connection failed: {e}")
        return {"status": "error", "detail": str(e)}


def _check_redis() -> dict:
    """同步检查 Redis 连接"""
    try:
        import redis
        client = redis.Redis(
            host=REDIS_CONFIG["REDIS_HOST"],
            port=REDIS_CONFIG["REDIS_PORT"],
            db=REDIS_CONFIG["REDIS_DB"],
            socket_connect_timeout=3,
        )
        client.ping()
        client.close()
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"Health check - Redis connection failed: {e}")
        return {"status": "error", "detail": str(e)}


async def _run_checks() -> dict:
    """在线程池中并发执行 MySQL 和 Redis 检查"""
    loop = asyncio.get_event_loop()
    mysql_result, redis_result = await asyncio.gather(
        loop.run_in_executor(_thread_pool, _check_mysql),
        loop.run_in_executor(_thread_pool, _check_redis),
    )
    return {"mysql": mysql_result, "redis": redis_result}


# 使用 lifespan 事件管理应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service_ready
    logger.info("Starting background task scheduler...")
    task1 = asyncio.create_task(building_bundle_of_index_task_scheduler())
    task2 = asyncio.create_task(building_stand_alone_index_task_scheduler())
    _service_ready = True
    logger.info("Service is ready.")
    try:
        yield
    finally:
        _service_ready = False
        logger.info("Shutting down background tasks...")
        task1.cancel()
        task2.cancel()
        try:
            await asyncio.gather(task1, task2, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        logger.info("Task scheduler stopped.")


app = FastAPI(lifespan=lifespan)

# 允许的来源通过 CORS_ALLOW_ORIGINS 配置（逗号分隔）。通配符与 credentials 不能同时启用。
CORS_ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()]
CORS_ALLOW_CREDENTIALS = "*" not in CORS_ALLOW_ORIGINS
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOW_ORIGINS, allow_credentials=CORS_ALLOW_CREDENTIALS,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(UserContextMiddleware)

# Include api_routers
app.include_router(llm.router)
app.include_router(local_knowledge_base_actions.router)
app.include_router(plugins.router)
app.include_router(prompt.router)
app.include_router(rag_checker.router)
app.include_router(mcp.router)

add_exception_handlers(app)


@app.get("/health/liveness", tags=["health"])
async def liveness():
    """存活探针：检查服务是否存活（含 MySQL、Redis 连接及服务启动状态）"""
    checks = {}
    is_alive = True

    # 检查 uvicorn 服务是否启动完毕
    if not _service_ready:
        checks["service"] = {"status": "error", "detail": "Service is not ready yet"}
        is_alive = False
    else:
        checks["service"] = {"status": "ok"}

    # 检查 MySQL 和 Redis
    middleware_checks = await _run_checks()
    checks.update(middleware_checks)

    if any(v.get("status") != "ok" for v in checks.values()):
        is_alive = False

    return JSONResponse(
        status_code=200 if is_alive else 503,
        content={"status": "alive" if is_alive else "unhealthy", "checks": checks},
    )


@app.get("/health/readiness", tags=["health"])
async def readiness():
    """就绪探针：检查服务是否就绪（含 MySQL、Redis 连接及服务启动状态）"""
    checks = {}
    is_ready = True

    # 检查 uvicorn 服务是否启动完毕
    if not _service_ready:
        checks["service"] = {"status": "error", "detail": "Service is not ready yet"}
        is_ready = False
    else:
        checks["service"] = {"status": "ok"}

    # 检查 MySQL 和 Redis
    middleware_checks = await _run_checks()
    checks.update(middleware_checks)

    if any(v.get("status") != "ok" for v in checks.values()):
        is_ready = False

    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": checks},
    )
