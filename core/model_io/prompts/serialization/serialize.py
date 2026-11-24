import json
import logging
import os
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate

from configs.model_config import DEFAULT_SERIALIZED_CN_PROMPT_TEMPLATE, DEFAULT_SERIALIZED_EN_PROMPT_TEMPLATE
from utils.common_util import random_string, remove_none_value_from_dict

logging.basicConfig()
logger = logging.getLogger(__name__)


class BaseSerializer:
    def __init__(self, save_dir: str, **kwargs):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def serialize(self,
                  prompt_type: str,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  have_context: bool = True,
                  **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class DefaultTypeSerializer(BaseSerializer):
    def serialize(self,
                  prompt_type: str,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  have_context: bool = True,
                  **kwargs) -> Dict[str, Any]:
        assert prompt_info_dict.get("template", None) is not None, "template str should not be None"
        input_variables = prompt_info_dict.get("input_variables", [])
        if have_context and "context" not in input_variables:
            logger.warning(f"input_variables should contain 'context' variable, but got {input_variables}")
            raise ValueError(f"input_variables should contain 'context' variable, but got {input_variables}")
        if "query" not in input_variables:
            logger.warning(f"input_variables should contain 'query', variable, but got {input_variables}")
            raise ValueError(f"input_variables should contain 'query', variable, but got {input_variables}")
        serialized_info = {
            "_type": "prompt",
            "input_variables": input_variables,
            "template": prompt_info_dict.get("template", ""),
        }
        try:
            PromptTemplate(**serialized_info, validate_template=True)
        except Exception as e:
            logger.warning(
                f"Prompt template validation failed. The prompt:{str(prompt_info_dict)}. Throw the error:{str(e)}")
            raise e
        return serialized_info


class ChatPromptTypeSerializer(BaseSerializer):
    def serialize(self,
                  prompt_type: str,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  have_context: bool = True,
                  **kwargs) -> Dict[str, Any]:
        """
        :param prompt_info_dict example:
        {
            "system":{
                "template":"你是一个AI对话助手。请用中文回答用户的问题，你的回答需要基于给定的上下文和对话历史，如果你不知道答案，请不要编造答案。",
            },
            "user":{
                "input_variables":[
                    "context",
                    "query",
                    "chat_history"
                ],
                "template":"以下是给定的上下文：\n {context}\n\n以下是对话历史：\n {chat_history}\n\n用户的问题是：{query}\n"
            }
        }
        """
        # 目前不再前端传递需要 user prompt info了
        # assert prompt_info_dict.get("user", None) is not None, "user node should not be None"
        system_info_dict = prompt_info_dict.get("system", None)
        if system_info_dict is None:
            system_info_dict = {
                "_type": "prompt",
                "template": DEFAULT_SERIALIZED_EN_PROMPT_TEMPLATE
            }
        else:
            system_info_dict["_type"] = "prompt"
        try:
            PromptTemplate(**system_info_dict, validate_template=True)
        except Exception as e:
            logger.warning(
                f"Prompt template validation failed. The prompt:{str(prompt_info_dict)}. Throw the error:{str(e)}")
            raise e

        # user_info_dict = prompt_info_dict.get("user", None)
        # user_info_dict["_type"] = "prompt"
        # input_variables = user_info_dict.get("input_variables", [])
        # if have_context and "context" not in input_variables:
        #     logger.warning(f"input_variables should contain 'context' variable, but got {input_variables}")
        #     raise ValueError(f"input_variables should contain 'context' variable, but got {input_variables}")
        # if "query" not in input_variables:
        #     logger.warning(f"input_variables should contain 'query', variable, but got {input_variables}")
        #     raise ValueError(f"input_variables should contain 'query', variable, but got {input_variables}")
        # try:
        #     PromptTemplate(**user_info_dict, validate_template=True)
        # except Exception as e:
        #     logger.warning(
        #         f"Prompt template validation failed. The prompt:{str(prompt_info_dict)}. Throw the error:{str(e)}")
        #     raise e
        serialized_info = {
            "system": system_info_dict,
            # "user": user_info_dict
        }
        return serialized_info


class FewShotTypeSerializer(BaseSerializer):
    def serialize(self,
                  prompt_type: str,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  have_context: bool = True,
                  **kwargs) -> Dict[str, Any]:
        assert prompt_info_dict.get("prefix", None) is not None, "prefix str should not be None"
        assert prompt_info_dict.get("suffix", None) is not None, "suffix str should not be None"
        input_variables = prompt_info_dict.get("input_variables", [])
        if have_context and "context" not in input_variables:
            logger.warning(f"input_variables should contain 'context' variable, but got {input_variables}")
            raise ValueError(f"input_variables should contain 'context' variable, but got {input_variables}")
        if "query" not in input_variables or "chat_history" not in input_variables:
            logger.warning(f"input_variables should contain 'query','chat_history' variable, but got {input_variables}")
            raise ValueError(
                f"input_variables should contain 'query','chat_history' variable, but got {input_variables}")

        example_prompt_info = None
        if prompt_info_dict.get("example_prompt_template", None):
            example_prompt_info = {
                "_type": "prompt",
                "input_variables": prompt_info_dict.get("example_prompt_input_variables", None),
                "template": prompt_info_dict.get("example_prompt_template", None)
            }
        examples = prompt_info_dict.get("examples", [])
        examples_file_path = os.path.join(self.save_dir, file_name + "_examples.json")
        with open(examples_file_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=4)

        serialized_info = {
            "_type": "few_shot",
            "prefix": prompt_info_dict.get("prefix", None),
            "input_variables": prompt_info_dict.get("input_variables", input_variables),
            "example_prompt": example_prompt_info,
            "examples": examples_file_path,
            "suffix": prompt_info_dict.get("suffix", None),
        }
        try:
            PromptTemplate(**example_prompt_info, validate_template=True)
        except Exception as e:
            logger.warning(
                f"Prompt template validation failed. The prompt:{str(prompt_info_dict)}. Throw the error:{str(e)}")
            raise e
        return serialized_info


class DefaultSerializePrompts:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def serialize(self,
                  prompt_type: str,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  have_context: bool = True,
                  **kwargs) -> Dict[str, Any]:
        '''
            It should be note that:
            if prompt_type is "default_prompt", **the input_variables must contains "context" variable**.
                It will be use to generate the context from retrievers for the prompt.
            If prompt_type is "few_shot_prompt", the "context" variable in input_variables is optional.
            And prompt_type should have "query" variable in input_variables, it will be use to generate the query.
            Example 1. For default_prompt:
                {
                    "_type": "prompt",
                    "input_variables": ["adjective", "content"],
                    "template": "Tell me a {adjective} joke about {content}."
                }

            Here is few_shot_prompt official doc: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples
            Example 2. For few_shot_prompt:
                {
                    "_type": "few_shot",
                    "input_variables": ["adjective"],
                    "prefix": "Write antonyms for the following words.",
                    "example_prompt": {
                        "_type": "prompt",
                        "input_variables": ["input", "output"],
                        "template": "Input: {input}\nOutput: {output}"
                    },
                    "examples": [
                        {"input": "happy", "output": "sad"},
                        {"input": "tall", "output": "short"}
                    ],
                    "suffix": "Input: {adjective}\nOutput:"
                }
            Load this prompy:
                    `Write antonyms for the following words.

                    Input: happy
                    Output: sad

                    Input: tall
                    Output: short

                    Input: funny
                    Output:    `

            so `examples` params in example_prompt_info_dict should be a List[json_str]

            :param prompt_type: "default_prompt" or "few_shot_prompt"
            :param prompt_info_dict:
            return: serialized_info
        '''
        if file_name is None:
            file_name = random_string(6) + "-" + prompt_type

        serializer: BaseSerializer = None
        if prompt_type == "default_prompt":
            serializer = DefaultTypeSerializer(save_dir=self.save_dir)
        elif prompt_type == "chat_prompt":
            serializer = ChatPromptTypeSerializer(save_dir=self.save_dir)
        elif prompt_type == "few_shot_prompt":
            serializer = FewShotTypeSerializer(save_dir=self.save_dir)

        serialized_info = serializer.serialize(prompt_type=prompt_type,
                                              prompt_info_dict=prompt_info_dict,
                                              file_name=file_name,
                                              have_context=have_context,
                                              **kwargs)

        serialized_info = remove_none_value_from_dict(serialized_info)
        abs_file_path = os.path.join(self.save_dir, file_name + ".json")
        # 暂时不需要保存
        # with open(abs_file_path, "w", encoding="utf-8") as f:
        #     json.dump(serialized_info, f, ensure_ascii=False, indent=4)

        resp = {
            "haveContext": have_context,
            "fileName": file_name,
            "absFilePath": abs_file_path,
            "serializeInfo": serialized_info,
            "serializedType": "default"
        }
        logger.info(f"Serialize prompt success. The prompt:{str(resp)}")
        return resp


class SerializeSystemMessageTemplatePrompts:
    '''
    暂不支持使用，勿用！
    '''

    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def serialize(self,
                  prompt_info_dict: Dict[str, Any] = {},
                  file_name: str = None,
                  **kwargs) -> Dict[str, Any]:
        system_prompt = prompt_info_dict.get("system_prompt", None)
        if system_prompt is None:
            return None

        if file_name is None:
            file_name = random_string(6) + "-message_template_prompt"

        serialized_info = {"system_prompt": system_prompt}
        # 暂时不需要保存
        # with open(os.path.join(self.save_dir, file_name + ".json"), "w", encoding="utf-8") as f:
        #     json.dump(serialized_info, f, ensure_ascii=False, indent=4)

        return {
            "fileName": file_name,
            "absFilePath": os.path.join(self.save_dir, file_name + ".json"),
            "serializedInfo": serialized_info,
            "serializedType": "system_message"
        }
