from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from prompt_definitions import PromptDefinition
from utils.lang_util import lang_def_name


class NerPrompt:
    @classmethod
    def get_openie_post_ner_prompts(cls, is_chinese_text=True) -> ChatPromptTemplate:
        prompt_definitions = PromptDefinition(lang=lang_def_name(is_chinese_text))
        openie_post_ner_prompts_json_dict = prompt_definitions.load_prompt_json_dict("openie_post_ner_prompt")
        openie_post_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(openie_post_ner_prompts_json_dict["openie_post_ner_instruction_system_prompt"]),
                                                                    HumanMessagePromptTemplate.from_template(openie_post_ner_prompts_json_dict["openie_post_ner_frame"])])
        return openie_post_ner_prompts

    @classmethod
    def get_query_ner_prompts(cls, is_chinese_text=True) -> ChatPromptTemplate:
        prompt_definitions = PromptDefinition(lang=lang_def_name(is_chinese_text))
        query_ner_prompts_json_dict = prompt_definitions.load_prompt_json_dict("query_ner_prompt")
        query_ner_prompts = ChatPromptTemplate.from_messages(
            [SystemMessage(query_ner_prompts_json_dict["system_message"]),
             HumanMessage(query_ner_prompts_json_dict["query_prompt_one_shot_input_human_prompt"]),
             AIMessage(query_ner_prompts_json_dict["query_prompt_one_shot_output_AI_prompt"]),
             HumanMessagePromptTemplate.from_template(query_ner_prompts_json_dict["query_prompt_template"])])
        return query_ner_prompts