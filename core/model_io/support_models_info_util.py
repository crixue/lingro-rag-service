from typing import Optional

from pydantic import BaseModel

from configs.model_config import SUPPORTED_MODELS


class SupportedModelsInfoUtil(BaseModel):

    @staticmethod
    def get_supported_models_info() -> list:
        return SUPPORTED_MODELS

    @staticmethod
    def find_supported_model_by_name(model_name: Optional[str] = "", model_alias_name: Optional[str] = "") -> dict:
        for model in SUPPORTED_MODELS:
            if model.get("name").lower() == model_name.lower():
                return model
            if model.get("alias").lower() == model_alias_name.lower():
                return model
        default_model = SUPPORTED_MODELS[0].get("name")
        for model in SUPPORTED_MODELS:
            if model.get("default", False):
                default_model = model
        return default_model

    @staticmethod
    def find_supported_model_by_id(model_id: str) -> dict:
        if model_id is not None:
            for model in SUPPORTED_MODELS:
                if model.get("id") == model_id:
                    return model
        for model in SUPPORTED_MODELS:
            if model.get("default", False):
                return model
        return SUPPORTED_MODELS[0]

    @staticmethod
    def default_build_kg_index_product_id() -> str | None:
        for model in SUPPORTED_MODELS:
            if model.get("default_build_knowledge_product_id", False):
                default_model = model
                return default_model.get("id")
        return None
