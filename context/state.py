from typing import Optional

from langchain.agents import AgentState

from controllers.vo.startup_llm_model_request_vo import RetrieverConfig, ChatSessionState


class ChatState(AgentState):
    retriever_config: Optional[RetrieverConfig]
    chat_session_state: Optional[ChatSessionState]
    context_tokens_limit: Optional[int]
    retriever_tool_definition: Optional[str]