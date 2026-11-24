from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models import BaseChatModel


@dataclass
class ApplicationContext:
    user_id: str
    llm_client: Optional[BaseChatModel]