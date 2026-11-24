import redis

from configs.model_config import logger, REDIS_CONFIG

# 连接/读写超时，避免 Redis 不可达时请求长时间挂起
REDIS_SOCKET_CONNECT_TIMEOUT = 3
REDIS_SOCKET_TIMEOUT = 5


class DefaultRedisClient:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_CONFIG["REDIS_HOST"], port=REDIS_CONFIG["REDIS_PORT"], db=REDIS_CONFIG["REDIS_DB"],
                             socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                             socket_timeout=REDIS_SOCKET_TIMEOUT)

    def get_redis_client(self):
        logger.info(f"[get_redis_client]-Connecting to Redis at {REDIS_CONFIG['REDIS_HOST']}:{REDIS_CONFIG['REDIS_PORT']}/db{REDIS_CONFIG['REDIS_DB']}")
        return self.r
