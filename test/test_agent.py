import asyncio
import json
import os
import unittest
from typing import Any

from api_routers.llm import find_model_info
from configs.model_config import logger, BASE_DIR
from controllers.vo.startup_llm_model_request_vo import RetrieverConfig, ChatRequestVO
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from service.chat_agent_service import DefaultChatAgentService
from tools_definitions.tools.builtin_tools import RETRIEVER_DOCS_TOOL_DEFAULT_NAME

class TestAgent(unittest.IsolatedAsyncioTestCase):
    async def test_agent_completions(self):
        with open(r"./startup_chat_request.json", "r") as f:
            chat_request = ChatRequestVO.model_validate_json(f.read())

        init_model_request, model_info = find_model_info(chat_request)
        langchain_llm = DefaultLangchainLLM(model_name=model_info.get("name"), user_id='unit-test',
                                            product_id=model_info.get("id"),
                                            model_config=init_model_request.llm_model_config)

        agent_service = DefaultChatAgentService(langchain_llm=langchain_llm)
        resp = await agent_service.completions(chat_request=chat_request)
        print(json.dumps(resp.dict(), indent=4, ensure_ascii=False))


def test0():
    def _special_tool_input_parse(tool_name: str, tool_params_input: str) -> Any:
        tool_params_input = tool_params_input.strip('"')
        try:
            tool_params_input = json.loads(tool_params_input)
            if tool_name == RETRIEVER_DOCS_TOOL_DEFAULT_NAME and "query" not in tool_params_input:
                # find first key's value whatever it is and use it as query
                tool_params_input = {"query": list(tool_params_input.values())[0]}
        except Exception as e:
            logger.warning(f"Failed to parse tool_input: {e}, will use raw tool_input")
        return tool_params_input

    result = _special_tool_input_parse(RETRIEVER_DOCS_TOOL_DEFAULT_NAME, '{"keywords": "成都天气"}')
    print(result)



def test_agent_completions_stream():
    async def run_async(agent_service: DefaultChatAgentService, chat_request: ChatRequestVO):
        async for chunk in agent_service.completions_stream(chat_request=chat_request):
            print(f"[completions_stream]-chunk output:{chunk}", end="", flush=True)
            # pass

    with open(os.path.join(BASE_DIR, "test", "startup_chat_request_stream.json"), "r") as f:
        json_str = json.load(f)
        chat_request = ChatRequestVO.parse_obj(json_str)

    init_model_request = chat_request.used_model_config
    langchain_llm = DefaultLangchainLLM(llm_type=init_model_request.llm_model_name,
                                     model_config=init_model_request.llm_model_config)
    agent_service = DefaultChatAgentService(langchain_llm=langchain_llm)
    asyncio.run(run_async(agent_service=agent_service, chat_request=chat_request))


# class TestAgentService(aiounittest.AsyncTestCase):
#     async def test_agent_completions_stream(self):
#         chat_request = None
#         with open(r"./startup_chat_request_stream.json", "r") as f:
#             json_str = json.load(f)
#             chat_request = ChatRequestVO.parse_obj(json_str)
#
#         init_model_request = chat_request.init_model_request
#         llm_client = DefaultLangchainLLM(llm_type=init_model_request.llm_type,
#                                         model_config=init_model_request.model_config)
#         agent_service = DefaultAgentService(llm_client=llm_client)
#         resp = agent_service.completions_stream(chat_request=chat_request)
#         for chunk in resp:
#             print(chunk)


def test_regrex():
    import re

    # text = ' 我应该使用get_current_datetime_f77b31 工具来获取当前的北京时间。\nAction: get_current_datetime_f77b31\nObserv'
    text = ' 我会尝试搜索关于"点读机女孩"和她与脑瘤相关的故事或新闻。\nAction: search_relevant_docs\nAction Input: {"query": "点读机女孩 脑瘤"}\nObserv'
    # text = ' 我会尝试搜索关于"点读机女孩"和她与脑瘤相关的故事或新闻。\nAction: search_relevant_docs\nObserv'

    ACTION_REGEX = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*(Action|Observ)"
    ACTION_INPUT_REGEX = r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(\{.*?\})"

    match = re.search(ACTION_REGEX, text)
    if match:
        action = match.group(1).strip()
        print(action)
        match_action_input = re.search(ACTION_INPUT_REGEX, text)
        if match_action_input:
            action_input = match_action_input.group(1)
        else:
            action_input = "{}"
        print(action_input)

