import json
import os
from typing import Optional

from langchain_core.prompts import BasePromptTemplate
from pydantic import Field, model_validator, BaseModel

from configs.model_config import logger, BASE_DIR
from core.model_io.prompts.serialization.deserialize import DefaultDeserializePrompts


class PromptDefinition(BaseModel):
    lang: str = Field(default="cn", examples=["cn", "en"])
    json_definition_dir: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    def root_validate(cls, values):
        lang = values.get("lang", None)
        if lang is None:
            logger.error("lang should not be None")
            raise ValueError("lang should not be None")
        tools_definitions_lang_path = os.path.join(BASE_DIR, "prompt_definitions", "builtin", "lang", values["lang"])
        if not os.path.exists(tools_definitions_lang_path):
            logger.error(f"lang {lang} not supported")
            raise ValueError(f"tools_definitions_lang_path {tools_definitions_lang_path} not exists")
        values["json_definition_dir"] = tools_definitions_lang_path
        return values

    def load_builtin_prompt(self, prompt_name: str) -> BasePromptTemplate:
        prompt_path = os.path.join(self.json_definition_dir, f"{prompt_name}.json")
        if not os.path.exists(prompt_path):
            logger.error(f"prompt_path {prompt_path} not exists")
            raise ValueError(f"prompt_path {prompt_path} not exists")
        return DefaultDeserializePrompts.load_prompt(prompt_path=prompt_path)

    def load_prompt_json_dict(self, prompt_name: str) -> dict:
        prompt_path = os.path.join(self.json_definition_dir, f"{prompt_name}.json")
        if not os.path.exists(prompt_path):
            logger.error(f"prompt_path {prompt_path} not exists")
            raise ValueError(f"prompt_path {prompt_path} not exists")
        with open(prompt_path, 'r', encoding="utf-8") as f:
            json_dict = json.load(f)
        return json_dict

    @classmethod
    def load_builtin_prompt_static(cls, lang: str, prompt_name: str, **kwargs) -> BasePromptTemplate:
        prompt_definition = PromptDefinition(lang=lang, json_definition_dir=kwargs.get("json_definition_dir", None))
        return prompt_definition.load_builtin_prompt(prompt_name=prompt_name)

    @classmethod
    def load_prompt_json_dict_static(cls, lang: str, prompt_name: str, **kwargs) -> dict:
        prompt_definition = PromptDefinition(lang=lang, json_definition_dir=kwargs.get("json_definition_dir", None))
        return prompt_definition.load_prompt_json_dict(prompt_name=prompt_name)