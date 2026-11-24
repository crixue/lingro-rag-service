import json
import re
from typing import Union, Any, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from pydantic import Field

from configs.model_config import logger
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import SEARCH_KEYWORDS_FLAG
from tools_definitions.tools.builtin_tools import RETRIEVER_DOCS_TOOL_DEFAULT_NAME

FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_CN_ACTION = "Final Answer："

MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

ACTION_REGEX = r"Action\s*\d*\s*[：|:][\s]*(.*?)[\s]*(Action|Observ)"
ACTION_INPUT_REGEX = r"Action\s*\d*\s*Input\s*\d*\s*[：|:][\s]*(\{.*?\})"


USED_FUNCTION_NAMES_FLAG = "used_function_names"

IS_FINAL_ANSWER = "is_final_answer"


class CustomReActSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """
    search_keywords: Optional[str] = Field(default=None)
    used_function_names: Optional[list[str]] = Field(default=[])

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def _special_tool_input_parse(self, tool_name: str, tool_params_input: str) -> Any:
        tool_params_input = tool_params_input.strip('"')
        try:
            tool_params_input = json.loads(tool_params_input)
            if tool_name == RETRIEVER_DOCS_TOOL_DEFAULT_NAME:
                if "query" not in tool_params_input:
                    # find first key's value whatever it is and use it as query
                    tool_params_input = {"query": list(tool_params_input.values())[0]}
                self.search_keywords = tool_params_input["query"]
        except Exception as e:
            logger.warning(f"Failed to parse tool_input: {e}, will use raw tool_input")
        return tool_params_input

    def _record_used_function_ids(self, tool_name: str):
        if tool_name == RETRIEVER_DOCS_TOOL_DEFAULT_NAME:
            return
        self.used_function_names.append(tool_name)

    def _extract_json_dict(text):
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
        match = re.search(pattern, text)

        if match:
            json_string = match.group()
            try:
                json_dict = json.loads(json_string)
                return json_dict
            except json.JSONDecodeError:
                return ''
        else:
            return ''

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text or FINAL_ANSWER_CN_ACTION in text
        action_match = re.search(ACTION_REGEX, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip().replace("[", "").replace("]", "")  # remove brackets， 解决 Action: [search_relevant_docs] 的情况
            action = action.replace("[", "").replace("]", "")
            match_action = re.search(ACTION_INPUT_REGEX, text)
            if match_action:
                action_input = match_action.group(1)
            else:
                action_input = "{}"
            tool_input = action_input.strip(" ")
            tool_input = self._special_tool_input_parse(action, tool_input)
            self._record_used_function_ids(action)

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return_values = {
                USED_FUNCTION_NAMES_FLAG: self.used_function_names,
                SEARCH_KEYWORDS_FLAG: self.search_keywords,
                "output": text.strip()
            }
            if FINAL_ANSWER_ACTION in text:
                return_values["output"] = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            elif FINAL_ANSWER_CN_ACTION in text:
                return_values["output"] = text.split(FINAL_ANSWER_CN_ACTION)[-1].strip()
            return AgentFinish(return_values, text)
        else:  # 可能存在llm没有返回 Final answer 也没有返回 Action 的情况, 这时候就直接返回 output当做最终答案
            return AgentFinish(
                {
                    IS_FINAL_ANSWER: True,
                    "output": text.strip(),
                    USED_FUNCTION_NAMES_FLAG: self.used_function_names,
                    SEARCH_KEYWORDS_FLAG: self.search_keywords
                }, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "react-single-input"


if __name__ == '__main__':
    text = '为了回答这个问题，我需要查找2011年到2018年间台积电股票的历史价格数据，以计算其涨幅。 Action: [search_relevant_docs] Action Input: {"query":"台积电 股票价格 2011年 2018年 历史数据"}'
    regex = r"Action\s*\d*\s*[：|:][\s]*(.*?)[\s]*(Action|Observ)"
    regex1 = r"Action\s*\d*\s*Input\s*\d*\s*[：|:][\s]*(\{.*?\})"
    # action_match = re.search(regex, text, re.DOTALL)
    # print(action_match.group(1))

    action_matches = re.findall(regex, text, re.DOTALL)
    actions = [match[0].strip() for match in action_matches]
    for action in actions:
        print(action)