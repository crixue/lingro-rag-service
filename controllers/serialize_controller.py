import os
from typing import Dict, Any

from configs.model_config import BASE_PROMPT_TEMPLATE_PATH
from core.model_io.prompts.serialization.serialize import DefaultSerializePrompts, SerializeSystemMessageTemplatePrompts


class BaseSerializeController:
    def __init__(self, **kwargs):
        pass

    def serialize(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class DefaultSerializeController(BaseSerializeController):
    def __init__(self, prompt_folder_name: str = None):
        super().__init__()
        assert prompt_folder_name is not None, "prompt_folder_name should not be None"

        self.save_dir = os.path.join(BASE_PROMPT_TEMPLATE_PATH, "default", prompt_folder_name)
        self.default_serialized_controller = DefaultSerializePrompts(save_dir=self.save_dir)

    def serialize(self, **kwargs) -> Dict[str, Any]:
        return self.default_serialized_controller.serialize(**kwargs)


class SystemMessageSerializeController(BaseSerializeController):
    def __init__(self, prompt_folder_name: str = None):
        super().__init__()
        self.save_dir = os.path.join(BASE_PROMPT_TEMPLATE_PATH, "system_messages", prompt_folder_name)
        self.serialized_controller = SerializeSystemMessageTemplatePrompts(save_dir=self.save_dir)

    def serialize(self, **kwargs) -> Dict[str, Any]:
        return self.serialized_controller.serialize(**kwargs)
