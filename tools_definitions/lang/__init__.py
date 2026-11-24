import os
import json
from typing import Any, Dict, Optional

from pydantic import Field, model_validator, BaseModel

from configs.model_config import logger, BASE_DIR
from utils.common_util import format_template_by_jinja2

def _render_obj(obj: Any, variables: Dict[str, Any]) -> Any:
    """
    递归渲染对象中所有的字符串字段（支持 dict/list/str/其他原子类型）
    """
    if isinstance(obj, str):
        try:
            return format_template_by_jinja2(obj, **variables)
        except Exception as e:
            logger.warning(f"Failed to render template string: {obj!r}, error: {e}")
            # 渲染失败时返回原始字符串，便于容错
            return obj
    elif isinstance(obj, dict):
        return {k: _render_obj(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_render_obj(i, variables) for i in obj]
    else:
        return obj

def format_tool_template_jinja(tool_json_path: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取 JSON 文件并对其中所有字符串字段使用 Jinja2 渲染。

    返回渲染后的 dict。
    """
    if not os.path.exists(tool_json_path):
        raise FileNotFoundError(f"Template file not found: {tool_json_path}")

    try:
        with open(tool_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load JSON from {tool_json_path}: {e}")
        raise

    rendered = _render_obj(data, variables)
    return rendered

def format_tool_template_jinja_for_key(tool_json_path: str, key: str, variables: Dict[str, Any]) -> Optional[Any]:
    """
    读取 JSON 文件并渲染后，返回指定 key 对应的值（如果存在）。
    key 为顶层键名；如果需要支持嵌套键，可以自行在外部解析或扩展。
    """
    rendered = format_tool_template_jinja(tool_json_path, variables)
    return rendered.get(key) if isinstance(rendered, dict) else None


class ToolSchemaDefinition(BaseModel):
    lang: str = Field(default="cn", examples=["cn", "en"])
    tool_definition_dir: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    def root_validate(cls, values):
        lang = values.get("lang", None)
        if lang is None:
            logger.error("lang should not be None")
            raise ValueError("lang should not be None")

        tool_definitions_lang_path = os.path.join(BASE_DIR, "tools_definitions", "lang", values["lang"])
        if not os.path.exists(tool_definitions_lang_path):
            logger.error(f"lang {lang} not supported")
            raise ValueError(f"tool_definitions_lang_path {tool_definitions_lang_path} not exists")

        values["tool_definition_dir"] = tool_definitions_lang_path
        return values

    def load_tool_schema_dict(self, tool_name: str, variables: Optional[Dict[str, Any]] = None) -> dict:
        """
        加载工具定义 JSON 文件，并使用 Jinja2 对字符串字段进行变量替换

        Args:
            tool_name: 工具名称（不含 .json 后缀）
            variables: 用于 Jinja2 模板渲染的变量字典

        Returns:
            渲染后的工具定义字典
        """
        tool_path = os.path.join(self.tool_definition_dir, f"{tool_name}.json")
        if not os.path.exists(tool_path):
            logger.error(f"tool_path {tool_path} not exists")
            raise ValueError(f"tool_path {tool_path} not exists")

        try:
            with open(tool_path, 'r', encoding="utf-8") as f:
                tool_dict = json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load JSON from {tool_path}: {e}")
            raise

        # 如果提供了变量，则进行模板渲染
        if variables:
            tool_dict = _render_obj(tool_dict, variables)

        return tool_dict

    @classmethod
    def load_builtin_tool_schema_dict_static(
        cls,
        lang: str,
        tool_name: str,
        variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> dict:
        """
        静态方法：加载内置工具定义并进行变量替换

        Args:
            lang: 语言代码
            tool_name: 工具名称
            variables: 模板变量
            **kwargs: 其他参数（如自定义 tool_definition_dir）
        """
        tool_schema_definition = ToolSchemaDefinition(
            lang=lang,
            tool_definition_dir=kwargs.get("tool_definition_dir", None)
        )
        return tool_schema_definition.load_tool_schema_dict(
            tool_name=tool_name,
            variables=variables
        )
