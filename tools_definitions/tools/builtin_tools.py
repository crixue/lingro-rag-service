import json
from typing import Type, Optional, List, Any

from langchain_core.agents import AgentFinish, AgentActionMessageLog
from langchain_core.callbacks import CallbackManagerForToolRun, BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from pydantic.v1 import validator

from configs.model_config import logger, DEFAULT_CONTEXT_TOKENS_LIMIT
from controllers.common.common_constant import MetadataConstants
from controllers.vo.startup_llm_model_request_vo import RetrieverConfig
from core.common.common_constant import SEARCH_RELEVANT_DOCS_TOOL_NAME
from core.data_connections.retrievers.default_retriever_wrapper_v2 import DefaultRetrieverV2
from utils.common_util import count_tokens

RETRIEVER_DOCS_TOOL_DEFAULT_NAME = SEARCH_RELEVANT_DOCS_TOOL_NAME


class CustomAgentResponse(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    metadata: Optional[dict] = Field(
        description="Relevant document metadata info",
        required=False,
    )


class SearchInput(BaseModel):
    query: str = Field(description="should be a search query", required=True)


class RetrieverDocsTool(BaseTool):
    """
    需要特别注意的是，如果使用的langchain < v3.0的版本， BaseTool使用的是Pydantic的v1兼容版本，所以其子类也需要使用Pydantic的v1兼容版本，例如@validator而不是@field_validator。
    如果后期升级到langchain v3.0以上的版本，BaseTool使用的是Pydantic的v2版本，那么其子类也需要使用Pydantic的v2版本。

    """
    name: str = SEARCH_RELEVANT_DOCS_TOOL_NAME
    description: str = "useful for when you need to answer questions about current events, you can search relevant information by this tool."
    args_schema: Type[BaseModel] = SearchInput
    response_format: str = "content_and_artifact"

    init_params: RetrieverConfig = Field(default=None)
    llm_client: BaseChatModel = Field(default=None)
    context_tokens_limit: int = DEFAULT_CONTEXT_TOKENS_LIMIT

    @property
    def tool_name(self) -> str:
        return self.name

    # @field_validator("llm_client")
    @validator("llm_client")
    def validate_llm_client(cls, field):
        if field is None:
            raise ValueError(f"id should not be None")
        return field

    @classmethod
    def from_retriever_docs_tool(cls,
                                 init_params: RetrieverConfig,
                                 llm_client: BaseChatModel,
                                 description: str = "useful for when you need to answer questions about current events, you can search relevant information by this tool.",
                                 context_tokens_limit: Optional[int] = None,
                                 callbacks: Optional[BaseCallbackHandler] = None
                                 ) -> "RetrieverDocsTool":
        if context_tokens_limit is None:
            context_tokens_limit = llm_client.context_tokens_limit

        return cls(
            name=RETRIEVER_DOCS_TOOL_DEFAULT_NAME,
            description=description,
            func=cls._run,
            init_params=init_params,
            llm_client=llm_client,
            context_tokens_limit=context_tokens_limit,
            callbacks=callbacks,
        )

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        def count_docs_token(documents: list[Document], additional_msg_count: int = 0) -> int:
            token_count = sum(
                count_tokens(json.dumps({'page_content': doc.page_content, }, ensure_ascii=False)) for doc in
                documents) + additional_msg_count
            logger.debug(
                f"[from_retriever_docs_tool]- context_tokens_limit:{self.context_tokens_limit}, token_count: {token_count}")
            return token_count

        num_docs = len(docs)
        if self.context_tokens_limit:
            while count_docs_token(docs) > self.context_tokens_limit:
                num_docs -= 1
                docs = docs[:num_docs]

        return docs

    @staticmethod
    def _wrap_doc_to_feed_to_llm(doc: Document) -> str:
        if doc is None or doc.page_content is None:
            return None

        feed_info = {
            "content": doc.page_content,
            "metadata": MetadataConstants.popup_useless_metadata(doc.metadata)
        }
        # Convert the metadata to a string representation
        feed_info_str = json.dumps(feed_info, ensure_ascii=False)
        return feed_info_str

    def _run(
            self,
            query: str,
            config: Optional[RunnableConfig] = None,
            **kwargs
    ) -> Any:
        # 从 config 中获取 user_id
        kwargs = kwargs or {}
        user_id = None
        if config and "configurable" in config:
            user_id = config["configurable"].get("user_id")

        # if self.init_params.retriever_type != BuildKnowledgeIndexTypeEnum.WEB_SEARCH:
        #     if self.init_params.knowledge_index_folder_id not in self.init_params.knowledge_index_folder_name:
        #         self.init_params.knowledge_index_folder_name = self.init_params.knowledge_index_folder_id + "/" + self.init_params.knowledge_index_folder_name
        retriever_docs_info: dict = {}  # {doc_id: Document}
        keywords: list[str] = []
        logger.info(
            f"[get_relevant_documents_from_RetrieverDocsTool]-Have Load Retriever config: {self.init_params.model_dump()}")
        try:
            custom_retriever = DefaultRetrieverV2(
                llm=self.llm_client,
                query_str=query,
                **self.init_params.model_dump(),
            )
            docs = custom_retriever.get_relevant_documents(query)
            keywords.extend(query)
        except Exception as e:
            logger.exception(f"[get_relevant_documents_from_retrievers]-Error in retriever: {e}")
            raise ToolException(f"The search_retriever_docs tool is not available.")

        find_docs = ""
        docs_total_tokens = 0
        if len(docs) > 0:  # loop doc and keep only page_content and metadata fields
            for index, doc in enumerate(docs):
                doc_info = self._wrap_doc_to_feed_to_llm(doc)
                if doc_info is None:
                    continue
                docs_total_tokens += count_tokens(doc_info)
                if docs_total_tokens > self.context_tokens_limit:
                    logger.info(
                        f"[get_relevant_documents_from_RetrieverDocsTool]- query:{query} have exceed context_tokens_limit:{self.context_tokens_limit}, docs_total_tokens: {docs_total_tokens}")
                    break
                find_docs += doc_info + "\n"
                if doc.id is not None:
                    retriever_docs_info[doc.id] = doc
        search_keywords = "".join(keywords)
        return find_docs, {
            "retriever_docs_info": retriever_docs_info.values(),
            "search_keywords": search_keywords,
            "retriever_docs_count": len(retriever_docs_info),
            "retriever_docs_total_tokens": docs_total_tokens
        }

