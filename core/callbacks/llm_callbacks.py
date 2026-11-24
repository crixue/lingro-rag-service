import asyncio
import json
from typing import Optional, Any
from uuid import UUID

import requests
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage, BaseMessageChunk
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from configs.model_config import logger, JAVA_MICRO_SERVICE_URL
from controllers.vo.model_payment_txn_request_vo import ModelPaymentTxnRequestVO
from exceptions.custom_exception import NotHaveEnoughMoneyException
from core.model_io.support_models_info_util import SupportedModelsInfoUtil


class BaseLLMTokenUsageCallbackFunction(AsyncCallbackHandler, BaseModel):
    """
        用于处理LLM模型使用之后的token使用情况，token的用量将发送给java端处理
    """
    product_type: str
    user_id: Optional[str] = Field(default=None)
    product_id: Optional[str] = Field(default=None)
    product_type: int = Field(default=1, description="1 for LLM, 5 for embedding")
    raise_error: bool = True

    @property
    def llm_type(self) -> str:
        return self.product_type

    def request_record_token_usage(self, prompt_tokens: int, completion_tokens: int, tokens: int,
                                   cache_creation_tokens: Optional[int] = None,
                                   cache_read_tokens: Optional[int] = None,
                                   request_id: Optional[str] = None, usage_metadata: Optional[str] = None) -> None:
        request_vo = ModelPaymentTxnRequestVO(
            userId=self.user_id,
            productId=self.product_id,
            productType=self.product_type,
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            cacheCreationTokens=cache_creation_tokens,
            cacheReadTokens=cache_read_tokens,
            totalTokens=tokens,
            thirdPartyId=request_id,
            usageMetadata=usage_metadata
        )

        try:
            requests.put(url=JAVA_MICRO_SERVICE_URL["count-user-token-usage"],
                         headers={"Content-Type": "application/json"},
                         json=request_vo.model_dump(by_alias=True, exclude_none=True))
            logger.debug(f"[count_user_tokens]-Counting user:{self.user_id} product:{self.product_id} prompt_tokens:{prompt_tokens} "
                        f"completion_tokens:{completion_tokens} total_tokens:{tokens} cached_read_tokens:{cache_read_tokens} "
                        f"cache_creation_tokens:{cache_creation_tokens} success")
        except Exception as e:
            logger.exception(
                f"Failed to count user:{self.user_id} product:{self.product_id} prompt_tokens:{prompt_tokens} "
                        f"completion_tokens:{completion_tokens} total_tokens:{tokens} with ex: {e}")

    async def request_record_token_usage_async(self, prompt_tokens: int, completion_tokens: int, tokens: int,
                                               cache_creation_tokens: Optional[int] = None,
                                               cache_read_tokens: Optional[int] = None,
                                               request_id: Optional[str] = None, usage_metadata: Optional[str] = None) -> None:
        self.request_record_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens=tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            request_id=request_id,
            usage_metadata=usage_metadata
        )

    async def count_user_tokens_async(self, message: BaseMessage | BaseMessageChunk,
                                      prompt_tokens_name: str = "input_tokens",
                                      completion_tokens_name: str = "output_tokens",
                                      token_name: str = "total_tokens",
                                      request_id_name: str = "id"):
        if not isinstance(message, AIMessage) and not isinstance(message, AIMessageChunk):
            return
        if ((usage_metadata := message.usage_metadata) is not None and
                usage_metadata.get(token_name, None) is not None):  # count tokens
            # count tokens
            logger.info(f"[count_user_tokens_async]- message id: {message.id}, usage_metadata: {usage_metadata}")
            cache_creation_tokens = None
            cache_read_tokens = None

            try:
                input_token_details = usage_metadata.get("input_token_details", {})
                if input_token_details:
                    cache_creation_tokens = input_token_details.get("cache_creation", 0) or None
                    cache_read_tokens = input_token_details.get("cache_read", 0) or None
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to parse usage_metadata: {usage_metadata}, error: {e}")

            await self.request_record_token_usage_async(
                prompt_tokens=int(usage_metadata.get(prompt_tokens_name, 0)),
                completion_tokens=int(usage_metadata.get(completion_tokens_name, 0)),
                tokens=int(usage_metadata.get(token_name, 0)),
                request_id=message.id,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                usage_metadata=json.dumps(usage_metadata, ensure_ascii=False, default=str)
            )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            message = response.generations[0][0].message
            await self.count_user_tokens_async(message)
        except Exception as e:
            logger.error(f"[on_llm_end]-Error in counting user tokens: {e}")
            pass

    def pre_check_have_enough_money(self,):
        if not self.user_id:
            raise NotHaveEnoughMoneyException()
        resp = requests.get(
            url=JAVA_MICRO_SERVICE_URL["check-user-has-enough-money"] + f"/{self.user_id}?productId={self.product_id}",
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200 or resp.json().get("code", -1) != 0:
            err_msg = resp.json().get("msg", -1)
            raise Exception(f"Error occurred do pre check: {err_msg}")
        result = resp.json()["data"]
        if not result:
            raise NotHaveEnoughMoneyException()

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            await self.pre_check_have_enough_money()
        except NotHaveEnoughMoneyException as e:
            logger.error(f"[on_llm_start]-Not have enough money for user: {self.user_id}, product: {self.product_id}")
            raise e

    async def on_chat_model_start(self,
                                  serialized: dict[str, Any],
                                  messages: list[list[BaseMessage]],
                                  *,
                                  run_id: UUID,
                                  parent_run_id: Optional[UUID] = None,
                                  tags: Optional[list[str]] = None,
                                  metadata: Optional[dict[str, Any]] = None,
                                  **kwargs: Any, ) -> Any:
        """
        Called when the chat model starts running.
        When the user has not enough money, we will raise an exception to cancel the request forcedly.
        """
        try:
            self.pre_check_have_enough_money()
        except NotHaveEnoughMoneyException as e:
            err_msg = str(f"Not have enough money for user: {self.user_id}, product: {self.product_id}")
            logger.error(f"[on_llm_start]-{err_msg}")
            future = asyncio.Future()
            future.set_exception(asyncio.CancelledError(f"{err_msg}"))
            return await future

class ZhipuTokenUsageCallbackFunction(BaseLLMTokenUsageCallbackFunction):
    async def count_user_tokens_async(self, message: AIMessageChunk,
                                      prompt_tokens_name: str = "prompt_tokens",
                                      completion_tokens_name: str = "completion_tokens",
                                      token_name: str = "total_tokens",
                                      request_id_name: str = "request_id"):
        if (message.response_metadata is not None and
                (usage_metadata := message.response_metadata.get("token_usage", {})) is not None):
            cache_creation_tokens = None
            cache_read_tokens = None
            try:
                input_token_details = usage_metadata.get("prompt_tokens_details", {})
                if input_token_details:
                    # cache_creation_tokens = input_token_details.get("cache_creation", 0) or None
                    cache_read_tokens = input_token_details.get("cached_tokens", 0) or None
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to parse usage_metadata: {usage_metadata}, error: {e}")

            # count tokens
            await self.request_record_token_usage_async(
                prompt_tokens=int(usage_metadata.get(prompt_tokens_name, 0)),
                completion_tokens=int(usage_metadata.get(completion_tokens_name, 0)),
                tokens=int(usage_metadata.get(token_name, 0)),
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                request_id=message.id,
                usage_metadata=json.dumps(usage_metadata, ensure_ascii=False, default=str)
            )


class OpenRouterGeminiTokenUsageCallbackFunction(BaseLLMTokenUsageCallbackFunction):
    async def count_user_tokens_async(self, message: AIMessageChunk,
                                      prompt_tokens_name: str = "input_tokens",
                                      completion_tokens_name: str = "output_tokens",
                                      token_name: str = "total_tokens",
                                      request_id_name: str = "request_id"):
        """
        暂时不支持gemini的cache token统计，因为openrouter没有返回gemini cache write相关数据
        """
        if (message.usage_metadata is not None and
                (usage_metadata := message.usage_metadata) is not None):
            cache_creation_tokens = None
            cache_read_tokens = None

            try:
                input_token_details = usage_metadata.get("input_token_details", {})
                if input_token_details:
                    pass
                    # cache_creation_tokens = input_token_details.get("cache_creation", 0) or None
                    # cache_read_tokens = input_token_details.get("cache_read", 0) or None
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to parse usage_metadata: {usage_metadata}, error: {e}")

            # count tokens
            await self.request_record_token_usage_async(
                prompt_tokens=int(usage_metadata.get(prompt_tokens_name, 0)),
                completion_tokens=int(usage_metadata.get(completion_tokens_name, 0)),
                tokens=int(usage_metadata.get(token_name, 0)),
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                request_id=message.id,
                usage_metadata=json.dumps(usage_metadata, ensure_ascii=False, default=str)
            )


class EmbeddingsTokenUsageCallbackFunction(BaseLLMTokenUsageCallbackFunction):
    pass
