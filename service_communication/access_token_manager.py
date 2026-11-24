import aiohttp
import time
import asyncio

from configs.model_config import AAD_INFO, logger, JAVA_MICRO_SERVICE_URL


class AccessTokenManager:
    def __init__(self):
        self.access_token = None
        self.expiry_time = 0  # 过期时间戳

    async def get_access_token_async(self):
        # 如果当前时间小于过期时间减去3分钟，直接返回内存中的token
        if self.access_token and time.time() < self.expiry_time - 180:
            return self.access_token

        async with aiohttp.ClientSession() as session:
            async with session.post(JAVA_MICRO_SERVICE_URL["request-client-access-token"], params=AAD_INFO) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data["access_token"]
                    # 计算过期时间戳
                    self.expiry_time = time.time() + int(data["expires_in"])
                    return self.access_token
                else:
                    logger.error(f"Failed to fetch access token: {response.status}, {await response.text()}")
                    raise Exception(f"Failed to fetch access token: {response.status}, {await response.text()}")
