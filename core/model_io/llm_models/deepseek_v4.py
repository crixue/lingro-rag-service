"""OpenAI chat wrapper."""

from __future__ import annotations

import logging
from typing import (
    Any,
    List,
    Optional, AsyncIterator, )

from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)


@DeprecationWarning
class ChatDeepSeek(ChatDeepSeek):

    def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("retriever_docs_tool", None)
        if payload.get("functions") is not None and len(payload.get("functions")) <= 0:
            payload.pop("functions", None)
        if payload.get("tools") is not None and (len(payload.get("tools")) <= 0):
            payload.pop("tools", None)
        else:
            formatted_tools = [
                convert_to_openai_tool(tool, strict=kwargs.get("strict", None)) for tool in kwargs.get("tools", [])
            ]
            payload["tools"] = formatted_tools
        print(f"ChatDeepSeek._get_request_payload: {payload}", flush=True)
        return payload

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                print(f"ChatDeepSeek._astream chunk: {chunk}", flush=True)
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                print(f"ChatDeepSeek._astream chunk: {chunk}", flush=True)
                yield chunk
