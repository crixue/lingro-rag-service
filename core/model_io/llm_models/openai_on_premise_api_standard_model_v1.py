import json
import logging
from typing import Dict, Any, List, Union, Literal, Collection, Optional, Iterator, Tuple, Callable, AsyncIterator

import requests
import sseclient
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk, LLMResult, Generation
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from pydantic import Field, model_validator
from requests import HTTPError, Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from configs.model_config import SUPPORTED_MODELS, DEFAULT_CONTEXT_TOKENS_LIMIT
from core.common.common_constant import TOTAL_TOKENS_KEY, REQUEST_ID_KEY
from tools_definitions import ToolsDefinition, execute_function_call
from utils.common_util import count_tokens

logger = logging.getLogger(__name__)


class OpenAIAPIStandardChat(BaseLLM):
    """OpenAI API standard chat LLM.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAIAPIStandardChat
            endpoint_url = (
                "http://"
    """
    request_url: str = "http://127.0.0.1:8000/"
    model_name: str = "standard-model-api-v1"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""
    streaming: bool = False
    """Whether to stream the results or not."""
    lang: str = "cn"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "standard-model-api-v1"

    @model_validator(mode='before')
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return self.model_kwargs

    def _get_chat_params(
        self, prompts: List[str], stop: Optional[List[str]] = None, messages: List[dict] = None
    ) -> Tuple:
        if len(prompts) > 1:
            raise ValueError(
                f"OpenAIChat currently only supports single prompt, got {prompts}"
            )
        if messages is None:
            messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params.get("max_tokens") == -1:
            # for ChatGPT api, omitting max_tokens is equivalent to having no limit
            del params["max_tokens"]
        return messages, params

    def _function_intercept(self,
                            full_response,
                            stream: bool,
                            messages: List,
                            prompts: List[str],
                            stop: Optional[List[str]] = None,
                            run_manager: Optional[CallbackManagerForLLMRun] = None,
                            **kwargs: Any,) -> Any:
        """Intercept and check if is function calls and execute them."""
        assistant_message = full_response["choices"][0].get("message", None)
        if assistant_message and assistant_message.get("function_call"):
            function_name = assistant_message["function_call"]["name"]
            try:
                execute_result = execute_function_call(assistant_message)

                used_function_ids = kwargs.get("used_function_ids", [])
                tools_definition = ToolsDefinition(lang=self.lang)
                used_function_ids.append(tools_definition.find_id_by_function_name(function_name))
            except Exception as e:
                logger.warning(
                    f"Failed to parse tool call, maybe the response is not a tool call or have been answered. Error: {e}")
                execute_result = f"Execute function failed by Error: {e}"

            # assistant_message['content'] = None
            if messages is None:
                messages = []
            messages.append(assistant_message)
            messages.append({"role": "function",
                             "name": function_name,
                             "content": execute_result})
            if stream:
                yield self._stream(prompts[0], stop, run_manager, messages=messages, **kwargs)
            else:
                return self._generate(prompts, stop, run_manager, messages=messages, **kwargs)
        return None

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        messages = kwargs.get("messages", None)
        messages, params = self._get_chat_params([prompt], stop, messages)
        messages = self._reduce_tokens_below_limit(messages)
        kwargs.pop("messages", None)

        payload = {**params, "messages": messages, "stream": True}
        print(f"payload: {json.dumps(payload, indent=4, ensure_ascii=False)}")
        llm_output: dict = {}
        generation_result_list: List[Generation] = []

        async for stream_resp in await completion_with_retry_event_stream_async(
            self.request_url, payload=payload, llm=self, run_manager=run_manager, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            assistant_message = stream_resp["choices"][0].get("message", None)
            if_callback_function_directly = kwargs.get("callback_function_directly", False)
            if assistant_message and if_callback_function_directly:
                llm_result = self._function_intercept(stream_resp, messages=messages, prompts=[prompt],
                                                      run_manager=run_manager,
                                                      **params)
                if llm_result is not None:
                    yield llm_result
            token = stream_resp["choices"][0]["delta"].get("content", "")
            finish_reason = stream_resp["choices"][0].get("finish_reason", None)
            if finish_reason == "stop":
                total_tokens = stream_resp["usage"].get("total_tokens", 1000)
                llm_output[TOTAL_TOKENS_KEY] = total_tokens
                llm_output[REQUEST_ID_KEY] = stream_resp.get("id", None)
            generation_chunk = GenerationChunk(text=token,
                            generation_info=dict(
                                finish_reason=stream_resp["choices"][0].get("finish_reason", None),
                                logprobs=stream_resp["choices"][0].get("logprobs", None),
                            ),
                            )
            generation_result_list.append(generation_chunk)
            yield generation_chunk

            if run_manager:
                # print(f"stream_resp: {stream_resp} \n")
                await run_manager.on_llm_new_token(stream_resp)

        if run_manager:
            await run_manager.on_llm_end(
                LLMResult(
                    generations=[generation_result_list],
                    llm_output=llm_output,
                ))

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        messages = kwargs.get("messages", None)
        messages, params = self._get_chat_params([prompt], stop, messages)
        messages = self._reduce_tokens_below_limit(messages)
        kwargs.pop("messages", None)

        payload = {**params, "messages": messages, "stream": True}
        print(f"payload: {json.dumps(payload, indent=4, ensure_ascii=False)}")

        llm_output: dict = {}
        generation_result_list: List[Generation] = []

        for chunk in completion_with_retry_event_stream(
            self.request_url, payload=payload, llm=self, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            assistant_message = choice.get("message", None)
            if_callback_function_directly = kwargs.get("callback_function_directly", False)
            if assistant_message and if_callback_function_directly:
                llm_result = self._function_intercept(chunk, messages=messages, prompts=[prompt],
                                                      run_manager=run_manager,
                                                      **params)
                if llm_result is not None:
                    return llm_result
            if choice["delta"] is None:
                continue
            token = choice["delta"].get("content", "")
            finish_reason = choice.get("finish_reason", None)
            if finish_reason == "stop":
                total_tokens = chunk["usage"].get("total_tokens", 1000)
                llm_output[TOTAL_TOKENS_KEY] = total_tokens
                llm_output[REQUEST_ID_KEY] = chunk.get("id", None)
            generation = GenerationChunk(text=token,
                            generation_info=dict(
                                finish_reason=choice.get("finish_reason", None),
                                logprobs=choice.get("logprobs", None),
                            ))
            generation_result_list.append(generation)
            yield generation

            if run_manager:
                # print(f"chunk: {chunk} \n")
                run_manager.on_llm_new_token(chunk)
        if run_manager:
            run_manager.on_llm_end(
                LLMResult(
                    generations=[generation_result_list],
                    llm_output=llm_output,))

    def _reduce_tokens_below_limit(self, messages: list[dict] or None) -> list[dict]:
        '''
        Reduce tokens below limit by removing first two messages - a pair of (user, assistant) each time
        '''
        def count_msg_token(msgs: list[dict], additional_msg_count: int = 0) -> int:
            token_count = sum(count_tokens(json.dumps(msg, ensure_ascii=False)) for msg in msgs) + additional_msg_count
            print(f"Request messages payload token_count: {token_count}")
            return token_count

        if messages is None:
            return []
        system_message = messages[0]
        if system_message["role"] != "system":
            logger.error(f"[_reduce_tokens_below_limit]-The first message should be system role's message!")
            raise ValueError("The first message should be system role's message!")
        # count system message's tokens
        system_message_token_count = count_tokens(json.dumps(system_message, ensure_ascii=False))

        messages = messages[1:]
        num_messages = len(messages)

        context_tokens_limit = DEFAULT_CONTEXT_TOKENS_LIMIT
        for model_param in SUPPORTED_MODELS.copy():
            if model_param.get("name", None) == self.model_name:
                context_tokens_limit = model_param.get("context_tokens_limit", DEFAULT_CONTEXT_TOKENS_LIMIT)
                break
        if context_tokens_limit:
            # remove first two messages - a pair of (user, assistant) each time
            while count_msg_token(messages, system_message_token_count) > context_tokens_limit:
                num_messages -= 2
                messages = messages[-num_messages:]

        messages = [system_message] + messages
        return messages

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
                    # yield chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])
        messages = kwargs.get("messages", None)
        messages, params = self._get_chat_params(prompts, stop, messages)
        messages = self._reduce_tokens_below_limit(messages)
        kwargs.pop("messages", None)

        kwargs.update({"used_function_ids": kwargs.get("used_function_ids", [])})
        params = {**params, **kwargs}
        payload = {**params, "messages": messages}
        print(f"payload: {json.dumps(payload, indent=4, ensure_ascii=False)}")

        full_response = completion_with_retry(
            request_url=self.request_url, payload=payload, llm=self, run_manager=run_manager, **params
        )
        # print(f"full_response status: {full_response}")
        # Because we use reAct agent to call function indirectly, so if you need to call function directly, you can set callback_function_directly as True
        if_callback_function_directly = kwargs.get("callback_function_directly", False)
        if if_callback_function_directly:
            llm_result = self._function_intercept(full_response, stream=False, messages=messages, prompts=prompts, run_manager=run_manager,
                                                  **params)
            if llm_result is not None:
                return llm_result
        llm_output = {
            "model_name": self.model_name,
        }

        if_parse_response = kwargs.get("parse_response", True)
        if not if_parse_response:
            if len(kwargs.get("used_function_ids", [])) > 0:
                full_response.update({"used_function_ids": kwargs.get("used_function_ids", [])})
            generation_result = json.dumps(full_response, ensure_ascii=False)
            return LLMResult(
                generations=[
                    [Generation(text=generation_result, )]
                ],
                llm_output=llm_output,
            )

        return LLMResult(
            generations=[
                [Generation(text=full_response["choices"][0]["message"]["content"],
                            generation_info=full_response)]
            ],
            llm_output=llm_output,
        )

def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response["choices"][0]["text"],
        generation_info=dict(
            finish_reason=stream_response["choices"][0].get("finish_reason", None),
            logprobs=stream_response["choices"][0].get("logprobs", None),
        ),
    )

def _update_response(response: Dict[str, Any], stream_response: Dict[str, Any]) -> None:
    """Update response from the stream response."""
    response["choices"][0]["text"] += stream_response["choices"][0]["text"]
    response["choices"][0]["finish_reason"] = stream_response["choices"][0].get(
        "finish_reason", None
    )
    response["choices"][0]["logprobs"] = stream_response["choices"][0]["logprobs"]


def _streaming_response_template() -> Dict[str, Any]:
    return {
        "choices": [
            {
                "text": "",
                "finish_reason": None,
                "logprobs": None,
            }
        ]
    }

def _create_retry_decorator(
    llm: OpenAIAPIStandardChat,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
        request_url: str,
        payload: Dict[str, Any],
        llm: OpenAIAPIStandardChat,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        headers = {"Content-Type": "application/json"}
        response: Response = requests.post(request_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed with response code: {response.status_code}\n and response message:{response.text}")
            raise HTTPError(f"Failed retrieve data from model!")

    return _completion_with_retry(**kwargs)


def completion_with_retry_event_stream(
        request_url: str,
        payload: Dict[str, Any],
        llm: OpenAIAPIStandardChat,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry_event_stream(**kwargs: Any) -> Any:
        headers = {"Content-Type": "application/json"}
        # logger.info(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        response: Response = requests.post(request_url, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data != "[DONE]":
                    data = json.loads(event.data)
                    yield data
        else:
            logger.error(f"Failed with response code: {response.status_code}\n and response message:{response.text}")
            raise HTTPError(f"Failed retrieve data from model!")

    return _completion_with_retry_event_stream(**kwargs)


async def completion_with_retry_event_stream_async(
        request_url: str,
        payload: Dict[str, Any],
        llm: OpenAIAPIStandardChat,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry_event_stream(**kwargs: Any) -> Any:
        headers = {"Content-Type": "application/json"}
        # logger.info(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        response: Response = requests.post(request_url, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data != "[DONE]":
                    data = json.loads(event.data)
                    yield data
        else:
            logger.error(f"Failed with response code: {response.status_code}\n and response message:{response.text}")
            raise HTTPError(f"Failed retrieve data from model!")

    return _completion_with_retry_event_stream(**kwargs)

async def acompletion_with_retry(
        request_url: str,
        payload: Dict[str, Any],
        llm: OpenAIAPIStandardChat,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        headers = {"Content-Type": "application/json"}
        response = requests.post(request_url, headers=headers, json=payload)
        return await response

    return await _completion_with_retry(**kwargs)


if __name__ == '__main__':
    llm = OpenAIAPIStandardChat(
        model="chatglm3-6b",
        request_url="http://127.0.0.1:11800/v1/chat/completions",
        max_tokens=2048,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.1,
        streaming=False
    )

    template = """{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    chain = prompt | llm
    result = chain.invoke({"question": "What is the capital of France?"})
    print(result)


