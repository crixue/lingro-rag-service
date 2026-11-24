from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import google.api_core
# TODO: remove ignore once the google package is published with types
import proto  # type: ignore[import]
from google.ai.generativelanguage_v1beta.types import (
    GenerateContentResponse,
)
from google.generativeai.caching import CachedContent  # type: ignore[import]
from google.generativeai.types import Tool as GoogleTool  # type: ignore[import]
from google.generativeai.types.content_types import (  # type: ignore[import]
    FunctionDeclarationType,
    ToolDict,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._common import (
    SafetySettingDict,
)
from langchain_google_genai._function_utils import (
    _ToolChoiceType,
    _ToolConfigDict,
)
from langchain_google_genai.chat_models import _response_to_result, ChatGoogleGenerativeAIError, logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.network_util import VPNContextManager


class ChatGoogleGenerativeAI(ChatGoogleGenerativeAI):

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        cached_content: Optional[str] = None,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
        )
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            generation_method=self.client.generate_content,
            metadata=self.default_metadata,
        )
        return _response_to_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        cached_content: Optional[str] = None,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.async_client:
            updated_kwargs = {
                **kwargs,
                **{
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                },
            }
            return await super()._agenerate(
                messages, stop, run_manager, **updated_kwargs
            )

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
        )
        response: GenerateContentResponse = _achat_with_retry(
            request=request,
            **kwargs,
            generation_method=self.async_client.generate_content,
            metadata=self.default_metadata,
        )
        return _response_to_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        cached_content: Optional[str] = None,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
        )
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            generation_method=self.client.stream_generate_content,
            **kwargs,
            metadata=self.default_metadata,
        )

        prev_usage_metadata: UsageMetadata | None = None
        for chunk in response:
            _chat_result = _response_to_result(
                chunk, stream=True, prev_usage=prev_usage_metadata
            )
            gen = cast(ChatGenerationChunk, _chat_result.generations[0])
            message = cast(AIMessageChunk, gen.message)

            curr_usage_metadata: UsageMetadata | dict[str, int] = (
                message.usage_metadata or {}
            )

            prev_usage_metadata = (
                message.usage_metadata
                if prev_usage_metadata is None
                else UsageMetadata(
                    input_tokens=prev_usage_metadata.get("input_tokens", 0)
                    + curr_usage_metadata.get("input_tokens", 0),
                    output_tokens=prev_usage_metadata.get("output_tokens", 0)
                    + curr_usage_metadata.get("output_tokens", 0),
                    total_tokens=prev_usage_metadata.get("total_tokens", 0)
                    + curr_usage_metadata.get("total_tokens", 0),
                )
            )

            if run_manager:
                run_manager.on_llm_new_token(gen.text)
            yield gen

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        cached_content: Optional[str] = None,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if not self.async_client:
            updated_kwargs = {
                **{
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                },
            }
            async for value in super()._astream(
                messages, stop, run_manager, **updated_kwargs
            ):
                yield value
        else:
            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content or self.cached_content,
                tool_choice=tool_choice,
            )

            logger.debug(f"[Google gemini _astream]-Request payload: {request}")
            prev_usage_metadata: UsageMetadata | None = None
            for chunk in _achat_with_retry(
                request=request,
                generation_method=self.async_client.stream_generate_content,
                metadata=self.default_metadata,
            ):
                _chat_result = _response_to_result(
                    chunk, stream=True, prev_usage=prev_usage_metadata
                )
                gen = cast(ChatGenerationChunk, _chat_result.generations[0])
                message = cast(AIMessageChunk, gen.message)

                curr_usage_metadata: UsageMetadata | dict[str, int] = (
                    message.usage_metadata or {}
                )

                prev_usage_metadata = (
                    message.usage_metadata
                    if prev_usage_metadata is None
                    else UsageMetadata(
                        input_tokens=prev_usage_metadata.get("input_tokens", 0)
                        + curr_usage_metadata.get("input_tokens", 0),
                        output_tokens=prev_usage_metadata.get("output_tokens", 0)
                        + curr_usage_metadata.get("output_tokens", 0),
                        total_tokens=prev_usage_metadata.get("total_tokens", 0)
                        + curr_usage_metadata.get("total_tokens", 0),
                    )
                )

                if run_manager:
                    await run_manager.on_llm_new_token(gen.text)
                logger.debug(f"[Google gemini _astream]-Response chunk: {gen}")
                yield gen


def _create_retry_decorator() -> Callable[[Any], Any]:
    """
    Creates and returns a preconfigured tenacity retry decorator.

    The retry decorator is configured to handle specific Google API exceptions
    such as ResourceExhausted and ServiceUnavailable. It uses an exponential
    backoff strategy for retries.

    Returns:
        Callable[[Any], Any]: A retry decorator configured for handling specific
        Google API exceptions.
    """
    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 2

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _chat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _chat_with_retry(**kwargs: Any) -> Any:
        with VPNContextManager():
            try:
                return generation_method(**kwargs)
            # Do not retry for these errors.
            except google.api_core.exceptions.FailedPrecondition as exc:
                if "location is not supported" in exc.message:
                    error_msg = (
                        "Your location is not supported by google-generativeai "
                        "at the moment. Try to use ChatVertexAI LLM from "
                        "langchain_google_vertexai."
                    )
                    raise ValueError(error_msg)

            except google.api_core.exceptions.InvalidArgument as e:
                raise ChatGoogleGenerativeAIError(
                    f"Invalid argument provided to Gemini: {e}"
                ) from e
            except Exception as e:
                raise e

    return _chat_with_retry(**kwargs)


def _achat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator()
    from google.api_core.exceptions import InvalidArgument  # type: ignore

    @retry_decorator
    def _achat_with_retry(**kwargs: Any) -> Any:
        with VPNContextManager():
            try:
                return generation_method(**kwargs)
            except InvalidArgument as e:
                # Do not retry for these errors.
                raise ChatGoogleGenerativeAIError(
                    f"Invalid argument provided to Gemini: {e}"
                ) from e
            except Exception as e:
                raise e

    return _achat_with_retry(**kwargs)

