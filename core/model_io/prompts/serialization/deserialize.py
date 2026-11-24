import json
import os.path
from pathlib import Path
from typing import Union, List, Dict

from langchain_core.prompts import load_prompt, BasePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_core.prompts.loading import _load_prompt

from utils.mysql_client import CustomMysqlClient


class DefaultDeserializePrompts:
    def __init__(self) -> None:
        pass

    @classmethod
    def load_prompt(cls, prompt_path: Union[str, Path]) -> BasePromptTemplate:
        """Load a prompt from a file."""
        return load_prompt(prompt_path)

@DeprecationWarning
class MysqlDeserializePrompts:
    def __init__(self):
        self.mysql_client = CustomMysqlClient()
        self.cursor = self.mysql_client.get_cursor()

    def load_prompt(self, prompt_id: str) -> Union[BasePromptTemplate, Dict[str, str]]:
        """Load a prompt from a file."""
        sql = """
            SELECT t.content as content, t.prompt_type as prompt_type FROM prompt t where t.id = %s
        """
        self.cursor.execute(sql, (prompt_id,))
        result = self.cursor.fetchone()
        self.mysql_client.close()
        if result is None:
            raise ValueError(f"Prompt not found: {prompt_id}")
        try:
            if result.get("prompt_type", "") == "default_prompt":
                prompt = _load_prompt(json.loads(result.get("content", "")))
                return prompt

            if result.get("prompt_type", "") == "chat_prompt":
                var0 = json.loads(result.get("content", ""))
                if var0 is None:
                    raise ValueError(f"Prompt not found: {prompt_id}")
                prompt = {
                    "system": var0["system"],
                    "user": var0["user"],
                }
                return prompt
        except Exception as e:
            raise ValueError(f"json load Prompt failed by id : {prompt_id}")
        return None


class DeserializeSystemMessageTemplatePrompts:
    '''
    https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/msg_prompt_templates
    '''
    def __init__(self) -> None:
        pass

    def load_prompt(self, prompt_path: Union[str, Path]) -> SystemMessagePromptTemplate:
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_dict = json.load(f)
            system_prompt = prompt_dict.get("system_prompt", "")
            return SystemMessagePromptTemplate.from_template(system_prompt)
