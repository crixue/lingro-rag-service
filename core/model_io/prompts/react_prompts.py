from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel

from prompt_definitions import PromptDefinition
from utils.lang_util import lang_def_name


class BaseReactPrompt(BaseModel):
    _model_start_name: str = "default"

    @property
    def model_start_name(self) -> str:
        return self._model_start_name

    @classmethod
    def get_react_system_prompt(cls,
                                have_context: bool = True,
                                is_chinese_text: bool = True,
                                have_plugins: bool = False,) -> BasePromptTemplate:
        system_prompt: BasePromptTemplate
        prompt_definitions = PromptDefinition(lang=lang_def_name(is_chinese_text))
        system_prompt = prompt_definitions.load_builtin_prompt("react_system_prompt_langchain_v3")
        # if have_context:
        #     system_prompt = prompt_definitions.load_builtin_prompt("react_system_prompt_langchain_v3")
        # else:
        #     if have_plugins:
        #         system_prompt = prompt_definitions.load_builtin_prompt("react_system_prompt_without_context")
        #     else:
        #         system_prompt = prompt_definitions.load_builtin_prompt(
        #             "react_system_prompt_without_context_without_plugins")
        return system_prompt
