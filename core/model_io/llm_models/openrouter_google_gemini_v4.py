"""OpenAI chat wrapper."""

from __future__ import annotations

import logging
from typing import (
    Any,
    List,
    Optional, Type, Dict,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from core.model_io.llm_models.llm_utils.message_tools_util import to_flat_tool_message

logger = logging.getLogger(__name__)

class ChatGoogleGemini(ChatOpenAI):
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

        # print(f"ChatGoogleGemini._get_request_payload: {payload}", flush=True)
        return payload

    def _convert_chunk_to_generation_chunk(
            self,
            chunk: dict,
            default_chunk_class: Type,
            base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        # print(f"ChatGoogleGemini._convert_chunk_to_generation_chunk: {chunk}", flush=True)
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )

        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if delta := top.get("delta", {}):
                    if reasoning_content := delta.get("reasoning_content"):
                        generation_chunk.message.additional_kwargs[
                            "reasoning_content"
                        ] = reasoning_content
                    elif reasoning_content := delta.get("reasoning"):
                        generation_chunk.message.additional_kwargs[
                            "reasoning_content"
                        ] = reasoning_content

                    # Handle tool calls
                    if tool_calls := delta.get("tool_calls"):
                        generation_chunk.message.tool_calls = []
                        for tool_call in tool_calls:
                            generation_chunk.message.tool_calls.append(
                                {
                                    "id": tool_call.get("id", ""),
                                    "type": "function",  # type: ignore
                                    "name": tool_call.get("function", {}).get(
                                        "name", ""
                                    ),
                                    "args": tool_call.get("function", {}).get(
                                        "arguments", ""
                                    ),
                                }
                            )

        return generation_chunk