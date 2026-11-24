import json
import os
from typing import Optional

from pydantic import Field, model_validator, BaseModel

from configs.model_config import logger, BASE_DIR


class JsonSchemaDefinition(BaseModel):
    lang: str = Field(default="cn", examples=["cn", "en"])
    json_definition_dir: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    def root_validate(cls, values):
        lang = values.get("lang", None)
        if lang is None:
            logger.error("lang should not be None")
            raise ValueError("lang should not be None")
        json_schema_definitions_lang_path = os.path.join(BASE_DIR, "json_schema_definitions", "builtin", "lang", values["lang"])
        if not os.path.exists(json_schema_definitions_lang_path):
            logger.error(f"lang {lang} not supported")
            raise ValueError(f"json_schema_definitions_lang_path {json_schema_definitions_lang_path} not exists")
        values["json_definition_dir"] = json_schema_definitions_lang_path
        return values


    def load_json_schema_dict(self, schema_name: str) -> dict:
        schema_path = os.path.join(self.json_definition_dir, f"{schema_name}.json")
        if not os.path.exists(schema_path):
            logger.error(f"schema_path {schema_path} not exists")
            raise ValueError(f"schema_path {schema_path} not exists")
        with open(schema_path, 'r', encoding="utf-8") as f:
            json_dict = json.load(f)
        return json_dict

    @classmethod
    def load_builtin_json_schema_dict_static(cls, lang: str, schema_name: str, **kwargs) -> dict:
        json_schema_definition = JsonSchemaDefinition(lang=lang, json_definition_dir=kwargs.get("json_schema_definitions", None))
        return json_schema_definition.load_json_schema_dict(schema_name=schema_name)

