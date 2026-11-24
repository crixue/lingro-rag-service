from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI

from core.model_io.llm_models.llm_utils.message_tools_util import to_flat_tool_message

logger = logging.getLogger(__name__)

class ChatGrokAI(ChatOpenAI):
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

        raw_tools = kwargs.get("tools", []) or []
        if not raw_tools:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
        else:
            payload["tools"] = [to_flat_tool_message(t) for t in raw_tools]
            payload["tool_choice"] = "auto"

        if payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}

        if self.model_kwargs is not None and self.model_kwargs.get("parallel_tool_calls") is not None:
            payload["parallel_tool_calls"] = self.model_kwargs.get("parallel_tool_calls")

        return payload
