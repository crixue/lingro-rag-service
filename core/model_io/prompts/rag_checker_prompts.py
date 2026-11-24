from langchain_core.prompts import BasePromptTemplate

from prompt_definitions import PromptDefinition
from utils.lang_util import lang_def_name


class RAGCheckerPrompts:
    @classmethod
    def get_rag_evaluator_prompts(cls,
                                  is_chinese_text: bool = True) -> BasePromptTemplate:
        system_prompt: BasePromptTemplate
        prompt_definitions = PromptDefinition(lang=lang_def_name(is_chinese_text))
        system_prompt = prompt_definitions.load_builtin_prompt("rag_evaluate_system_prompt")
        return system_prompt