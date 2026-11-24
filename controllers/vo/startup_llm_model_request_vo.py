from typing import Dict, Any, Union, Optional, Literal, List

from langchain_core.documents import Document
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel, Field, field_validator, model_validator

from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.vo.mcp_connection import MCPServer, MCPServerWithTools
from utils.common_util import detect_if_is_chinese_text


class UsedModelConfig(BaseModel):
    is_chinese_text: Optional[bool] = Field(default=True)
    have_context: bool = Field(default=True)
    llm_id: Optional[str] = Field(default=None)
    llm_model_name: Optional[str] = Field(default=None, alias="model_name")
    llm_model_config: Optional[Dict[str, Any]] = Field(default={}, alias="model_config")


class ChatMessage(ChatCompletionMessage):
    role: Literal["user", "assistant", "system", "function", "tool"]


class ChatCompletionRequest(BaseModel):
    few_shot_examples: list[dict] = Field(default=None, example=[{"input": "2+2", "output": "4"}])
    history_messages: list[ChatMessage] = Field(default=[], example=[{"role": "user", "content": "hello"}])
    query: str = Field(default=None)
    used_model_config: UsedModelConfig = Field(default=None)
    context_docs: list[Document] = Field(default=[])
    prompt_template_str: str = Field(default=None)
    used_functions: Optional[list[str]] = Field(default=None, description="Not support for stream mode!")

    @field_validator("query")
    def validate_query(cls, field):
        if field is None or field == "":
            raise ValueError(f"query should not be empty")
        return field


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = Field(default=None)


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = Field(default=None)


class ChatCompletionResponse(BaseModel):
    id: Optional[str] = Field(default=None)
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default=None)
    used_function_ids: Optional[List[str]] = Field(default=None)
    retriever_docs: Optional[List[Document]] = Field(default=None)


class RequestMessage(BaseModel):
    role: str
    content: str


class ChatSessionUploadFileInfo(BaseModel):
    chat_session_id: str = Field(default=None)
    file_sync_record_id: str = Field(default=None)
    file_name: str = Field(default=None)
    file_size: Optional[int] = Field(default=None)
    last_modified: Optional[str] = Field(default=None)
    is_from_extension: Optional[bool] = Field(default=False, description="是否来自于浏览器插件在页面上刮取的网页内容")
    file_metadata: Optional[Dict[str, Any]] = Field(default=None, description="文件的额外元信息")

class ChatSessionState(BaseModel):
    uploaded_files_info: Optional[Dict[str, ChatSessionUploadFileInfo]] = Field(default=None, description="key is file_sync_record_id")

class RetrieverConfig(BaseModel):
    is_chinese_text: Optional[bool] = Field(default=True)
    retriever_type: Union[BuildKnowledgeIndexTypeEnum, list[BuildKnowledgeIndexTypeEnum]] = Field(default="A01")
    knowledge_index_folder_id: Optional[str] = Field(default=None,
                                           description="用户文件夹id,如果retriever_type不为web_search,local_vs_folder_name=userFolderId/local_vs_folder_name")
    knowledge_index_folder_name: str = Field(default="web_search")
    search_type: str = Field(default="similarity")
    search_top_k: int = Field(default=5)
    version: Optional[int] = Field(default=-1)
    web_search_results_count: Optional[int] = Field(default=20)
    fixed_weights: list[float] = Field(default=[0.5, 0.5])
    ignore_content_len: int = Field(default=120)
    cn_chunk_size: Optional[int] = Field(default=None)
    cn_chunk_overlap: Optional[int] = Field(default=None)
    en_chunk_size: Optional[int] = Field(default=None)
    en_chunk_overlap: Optional[int] = Field(default=None)
    use_embedding_filter_assist: Optional[bool] = Field(default=False)
    use_reorder_assist: Optional[bool] = Field(default=False)

    @field_validator("fixed_weights")
    def validate_fixed_weights(cls, field):
        if sum(field) != 1:
            raise ValueError(f"fixed_weights should be sum to 1, but got {field}")
        return field

    @field_validator("retriever_type")
    def validate_retriever_type(cls, field):
        if field is None:
            raise ValueError(f"retriever_type should not be None")
        return field

    @field_validator("search_type")
    def validate_search_type(cls, field):
        if field not in ["similarity", "mmr"]:
            raise ValueError(f"search_type should be one of [similarity, mmr], but got {field}")
        return field

    @model_validator(mode='before')
    def root_validate(cls, values):
        if values.get("retriever_type") not in BuildKnowledgeIndexTypeEnum.values():
            raise ValueError(
                f"retriever_type should be one of {BuildKnowledgeIndexTypeEnum.values()}, but got {values.get('retriever_type')}")
        return values


def _detect_chinese_text(history_messages: List[ChatMessage], query: str) -> bool:
    detect_size = 8000
    truncated_content = ""
    for message in history_messages:
        if message and message.content:
            truncated_content += message.content
            if len(truncated_content) >= detect_size:
                break
    truncated_content += query
    truncated_content = truncated_content[:detect_size]
    return detect_if_is_chinese_text(truncated_content)


class BaseChatRequestVO(BaseModel):
    is_chinese_text: Optional[bool] = Field(default=True)
    history_messages: List[ChatMessage] = Field(default=list, example=[{"role": "user", "content": "hello"}])
    used_functions: Optional[List[str]] = Field(default=None)
    retriever_config: RetrieverConfig = Field(default=None)
    chat_session_upload_file_info: Optional[ChatSessionUploadFileInfo] = Field(default=None)
    used_model_config: UsedModelConfig = Field(default=None)
    used_mcp_tools: Optional[List[MCPServerWithTools]] = Field(default=None, description="MCP server with selected tools")
    id: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    def validate_before(cls, values):
        if values.get("history_messages", None) is None or len(values.get("history_messages", None)) == 0:
            raise ValueError("history_messages should contains at least a system message")
        if values.get("retriever_config", None) is None:
            raise ValueError("retriever_config should not be None")
        if values.get("used_model_config", None) is None:
            raise ValueError("used_model_config should not be None")
        return values

    @model_validator(mode='after')
    def validate_after(self):
        is_chinese_text = _detect_chinese_text(self.history_messages, "")
        self.retriever_config.is_chinese_text = is_chinese_text
        self.used_model_config.is_chinese_text = is_chinese_text
        return self


class ChatRequestVO(BaseChatRequestVO):
    query: str = Field(default=None, description="User input")


class ChatBatchRequestVO(BaseChatRequestVO):
    queries: List[str] = Field(default=None, description="User input")


class BaseChatResponseVO(BaseModel):
    id: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: Optional[List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]] = Field(default=None)
    created: Optional[int] = Field(default=None)
    index: Optional[int] = Field(default=None)
    finish_reason: Optional[str] = Field(default=None)
    tool_call: Optional[Dict[str, Any]] = Field(default=None)


class ChatResponseVO(BaseChatResponseVO):
    pass


class ChatStreamResponseVO(BaseChatResponseVO):
    content: Optional[str] = Field(default=None)
    type: Optional[Literal["content", "addition_info", "reasoning_content", "tool_call"]] = Field(default=None)

    retriever_docs: Optional[List[Document]] = Field(default=None)
    search_keywords: Optional[str] = Field(default=None)

    @classmethod
    def create_addition_info(cls, retriever_docs: Optional[List[Document]], search_keywords: Optional[str]):
        return cls(
            retriever_docs=retriever_docs,
            search_keywords=search_keywords,
            object="chat.completion.chunk",
            type="addition_info")

    @classmethod
    def create_content(cls,
                       content: str,
                       resp_object: Literal["chat.completion", "chat.completion.chunk"],
                       content_type: Optional[
                           Literal["content", "addition_info", "reasoning_content", "tool_call"]] = "content",
                       *,
                       id: Optional[str] = None,
                       **kwargs
                       ):
        return cls(content=content,
                   object=resp_object,
                   type=content_type,
                   id=id,
                   **kwargs)
