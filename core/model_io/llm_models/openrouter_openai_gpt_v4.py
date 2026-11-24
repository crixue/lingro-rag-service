"""OpenAI chat wrapper."""

from __future__ import annotations

import logging
from typing import (
    Any,
    List,
    Optional,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

@DeprecationWarning
class ChatOpenAI(ChatOpenAI):
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
        if payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}

        if self.model_kwargs is not None and self.model_kwargs.get("parallel_tool_calls") is not None:
            payload["parallel_tool_calls"] = self.model_kwargs.get("parallel_tool_calls")

        return payload