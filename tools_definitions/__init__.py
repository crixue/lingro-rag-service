from typing import Optional, Literal, Type
import glob
import os
from typing import List

from langchain_core.tools import StructuredTool

from configs.model_config import BASE_DIR, logger
from utils.common_util import remove_none_value_from_dict
from pydantic import Field, SecretStr, model_validator, BaseModel, create_model
from tools_definitions.tools.simple_tools import *
from .openai_standard_function_entity import OpenAIStandardFunctionBO
from .tools import *


def execute_function_call(assistant_message: dict):
    """
    json_string = '{"function_call":{"arguments":{"location":"Boston","unit":"celsius"},"name":"get_current_weather"}}'
    assistant_message = json.loads(json_string)
    """

    function_call = assistant_message["function_call"]
    if function_call is None:
        logger.error(f"Doesn't exist function_call.")
        raise Exception(f"Doesn't exist function_call.")
    function_name = function_call["name"]
    if function_name is None:
        logger.error(f"Doesn't exist function_name.")
        raise Exception(f"Doesn't exist function_name.")
    function_arguments = json.loads(function_call.get("arguments",'{}'))
    function = globals().get(function_name, None)
    if function is None:
        logger.error(f"Doesn't exist this function: {function_name}.")
        raise Exception(f"Doesn't exist this function: {function_name}.")
    else:
        try:
            results = function(**function_arguments)
            return results
        except Exception as e:
            logger.error(f"Call function {function_name} failed with these function name {json.dumps(**function_arguments, ensure_ascii=False)} failed, exception: {e}")
            raise Exception(f"Call function {function_name} failed with these function name {json.dumps(**function_arguments, ensure_ascii=False)} failed, exception: {e}")


def create_pydantic_args_model_from_json(json_dict: dict, model_name: str) -> Type[BaseModel]:
    """
    从 JSON 字符串动态创建 Pydantic 模型
    """
    def mapping_type(field_type_str: str):
        if field_type_str == 'string':
            return str
        elif field_type_str == 'number':
            return float
        elif field_type_str == 'boolean':
            return bool
        elif field_type_str == 'list':
            return list
        elif field_type_str == 'tuple':
            return tuple
        elif field_type_str == 'set':
            return set
        elif field_type_str == 'dict':
            return dict
        else:
            raise ValueError(f'不支持的字段类型: {field_type_str}')

    required_fields = json_dict['required']
    # 获取模型字段
    fields = {}
    for field_name, field_info in json_dict['properties'].items():
        field_type = mapping_type(field_info['type'])
        field_description = field_info.get('description', None)
        if 'enum' in field_info:
            field_enum = field_info['enum']
            fields[field_name] = (Literal[tuple(field_enum)], Field(description=field_description))
            if field_name not in required_fields:
                fields[field_name] = (Optional[Literal[tuple(field_enum)]], Field(default=None, description=field_description))
        else:
            if field_name not in required_fields:
                fields[field_name] = (Optional[field_type], Field(default=None, description=field_description))
            else:
                fields[field_name] = (field_type, Field(description=field_description, required=True))
    # 使用 create_model 方法创建 Pydantic 模型
    pydantic_model = create_model(model_name,
                                  __base__=BaseModel,
                                  **fields)
    # # 设置必填字段
    # for field_name in required_fields:
    #     pydantic_model.__fields__[field_name].required = True
    return pydantic_model


class ToolsDefinition(BaseModel):
    lang: str = Field(default="cn")
    json_definition_dir: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    def root_validate(cls, values):
        lang = values.get("lang", None)
        if lang is None:
            logger.error("lang should not be None")
            raise ValueError("lang should not be None")
        tools_definitions_lang_path = os.path.join(BASE_DIR, "tools_definitions", "lang", values["lang"])
        if not os.path.exists(tools_definitions_lang_path):
            logger.error(f"lang {lang} not supported")
            raise ValueError(f"tools_definitions_lang_path {tools_definitions_lang_path} not exists")
        values["json_definition_dir"] = tools_definitions_lang_path
        return values

    def list_all_supported_function_definitions(self,) -> List[dict]:
        func_def_jsons = glob.glob(f"{self.json_definition_dir}/*.json")
        supported_functions = []
        for json_file in func_def_jsons:
            var = json.load(open(json_file, "r", encoding="utf-8"))
            if var.get("status", 0) > 0:
                supported_functions.append(var)
        return supported_functions

    def find_id_by_function_name(self, function_name: str) -> str:
        """
        Find function id by function name.
        :param function_name:
        :return:
        """
        func_def_jsons = glob.glob(f"{self.json_definition_dir}/*.json")
        for json_file in func_def_jsons:
            var = json.load(open(json_file, "r", encoding="utf-8"))
            if var["name"] == function_name.strip():
                return var["id"]
        return None

    def find_all_function_definitions_by_id_list(self, function_id_list: List[str]) -> List[dict]:
        """
        Find all function definitions by function id list.
        :param function_id_list:
        :return:
        """
        func_def_jsons = glob.glob(f"{self.json_definition_dir}/*.json")
        supported_functions = []
        for json_file in func_def_jsons:
            var = json.load(open(json_file, "r", encoding="utf-8"))
            if var["id"] in function_id_list:
                supported_functions.append(remove_none_value_from_dict(var))
        return supported_functions

    def create_langchain_tool_by_id(self, function_id: str, return_direct: bool = False) -> StructuredTool or None:
        """
        Create langchain tool by function id.
        :param function_id:
        :return:
        """
        func_def_jsons = glob.glob(f"{self.json_definition_dir}/*.json")
        for json_file in func_def_jsons:
            var = json.load(open(json_file, "r", encoding="utf-8"))
            if var["id"] == function_id.strip():
                function_name = var["name"]
                assert function_name is not None, "function_name is None, please check json file"
                parameters_json = var["parameters"]
                function = globals().get(function_name, None)
                tool = StructuredTool(
                    func=function,
                    name=var["name"],
                    description=var["description"],
                    args_schema=create_pydantic_args_model_from_json(parameters_json, "ArgsDynamicModel"),
                    return_direct=return_direct,
                    # coroutine= ... <- you can specify an async method if desired as well
                )
                return tool
        return None

    def create_all_tools(self, return_direct: bool = False) -> List[StructuredTool]:
        tools = []
        func_def_jsons = glob.glob(f"{self.json_definition_dir}/*.json")
        for json_file in func_def_jsons:
            var = json.load(open(json_file, "r", encoding="utf-8"))
            function_name = var["name"]
            parameters_json = var["parameters"]
            function = globals().get(function_name, None)
            tool = StructuredTool.from_function(
                func=function,
                name=var["name"],
                description=var["description"],
                args_schema=create_pydantic_args_model_from_json(parameters_json, "ArgsDynamicModel"),
                return_direct=return_direct,
                # coroutine= ... <- you can specify an async method if desired as well
            )
            tools.append(tool)
        return tools