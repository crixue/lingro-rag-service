import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

from langchain.agents import create_agent
from langchain.agents.middleware.types import ResponseT, _OutputAgentState, _InputAgentState, AgentState
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT
from openai import RateLimitError
from pydantic import BaseModel
from mcp.types import Tool as MCPTool

from api_routers.context import get_user_id
from configs.model_config import AGENT_MAX_ITERATIONS, logger, DEFAULT_CONTEXT_TOKENS_LIMIT
from context.context import ApplicationContext
from context.state import ChatState
from controllers.common.common_constant import MetadataConstants
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.vo.mcp_connection import MCPServerWithTools
from controllers.vo.startup_llm_model_request_vo import ChatRequestVO, ChatMessage, ChatResponseVO, \
    ChatCompletionResponseChoice, ChatStreamResponseVO, ChatBatchRequestVO, RetrieverConfig, ChatSessionState
from core.common.common_constant import FALL_BACK_ANSWER, FALL_BACK_ANSWER_ZH, RATE_LIMIT_ERROR_MESSAGE, \
    RATE_LIMIT_ERROR_MESSAGE_ZH, SEARCH_RELEVANT_DOCS_TOOL_NAME
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from prompt_definitions import PromptDefinition
from service.chat_message_util import handle_ai_message_chunk, handle_tool_message, handle_tool_message_chunk, handle_function_message_chunk
from tools_definitions.mcp.mcp_tools import convert_connections
from tools_definitions.tools.data_time_tools import get_current_date_time_tool
from tools_definitions.tools.file_system_tools import read_file_content_from_knowledge_tool, \
    read_uploaded_file_content_tool
from tools_definitions.tools.knowledge_base_info_tools import can_use_get_build_task_files_detail_tool, \
    get_knowledge_base_files_detail_tool
from tools_definitions.tools.retriever_tools import search_relevant_documents_from_knowledge_base_tool, \
    create_search_relevant_documents_from_uploaded_files_tool
from tools_definitions.tools.summarize_tools import summarize_knowledge_base_file_tool, \
    create_summarize_chat_session_uploaded_file_tool
from utils.lang_util import lang_def_name


def get_retriever_tool_definition_prompt(is_chinese_text: bool,
                                         retrieve_type: BuildKnowledgeIndexTypeEnum = BuildKnowledgeIndexTypeEnum.WEB_SEARCH) -> str:
    definition_prompt_dict = PromptDefinition.load_prompt_json_dict_static(
        lang=lang_def_name(is_chinese_text=is_chinese_text),
        prompt_name="tool_definition_prompt")

    if lang_def_name(is_chinese_text) == "cn":
        definition = "这个工具可以检索相关信息。你可以生成一些关键词，然后通过这个工具来搜索相关信息。这个工具需要你传入一个名为 query 的参数，这个参数代表你生成的关键词。"
    else:
        definition = "This tool will retrieve relevant information. You can generate some keywords and search for relevant information through this tool."

    # 检索器类型到描述键名的映射
    retriever_type_to_prompt_key = {
        BuildKnowledgeIndexTypeEnum.WEB_SEARCH: "web_search_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.ELASTICSEARCH: "elastic_search_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.HIPPO_KNOWLEDGE_GRAPH: "knowledge_graph_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.LOCAL_VECTOR_STORE: "vector_store_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.MIXED_LOCAL_VECTOR_STORE_AND_WEB_SEARCH: "mixed_vector_store_and_web_search_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.MIXED_ELASTICSEARCH_AND_WEB_SEARCH: "mixed_elastic_search_and_web_search_retriever_tool_def",
        BuildKnowledgeIndexTypeEnum.MIXED_HIPPO_KNOWLEDGE_GRAPH_AND_WEB_SEARCH: "mixed_knowledge_graph_and_web_search_retriever_tool_def",
    }

    prompt_key = retriever_type_to_prompt_key.get(retrieve_type)
    if prompt_key and prompt_key in definition_prompt_dict:
        return definition_prompt_dict.get(prompt_key)

    return definition


class BaseChatAgentService(BaseModel, ABC):
    langchain_llm: DefaultLangchainLLM
    react_system_prompt: Optional[Union[BasePromptTemplate, str]] = None
    react_user_prompt: Optional[Union[BasePromptTemplate, str]] = None
    is_chinese_text: Optional[bool] = True

    retriever_config: Optional[RetrieverConfig] = None
    context: ApplicationContext = None
    context_tokens_limit: Optional[int] = DEFAULT_CONTEXT_TOKENS_LIMIT
    chat_session_state: Optional[ChatSessionState] = None
    fall_back_answer: str = FALL_BACK_ANSWER

    class Config:
        arbitrary_types_allowed = True

    @property
    def _supported_models(self) -> List[str]:
        return []

    @abstractmethod
    def completions_stream(self,
                           chat_request: ChatRequestVO,
                           **kwargs):
        raise NotImplementedError

    @abstractmethod
    def completions(self,
                    chat_request: ChatRequestVO,
                    **kwargs):
        raise NotImplementedError

    async def abatch_as_completed_completions(self,
                                              chat_batch_request: ChatBatchRequestVO,
                                              **kwargs):
        pass


def prepare_input_messages(history_messages: List[ChatMessage], user_input: str) -> List[dict]:
    messages = ([message.model_dump(exclude_none=True) for message in history_messages] +
                [{"role": "user", "content": user_input}])
    return messages


async def create_mcp_tools_async(mcp_tools: Optional[List[MCPServerWithTools]]) -> List[BaseTool] | None:
    if not mcp_tools or len(mcp_tools) <= 0:
        return None

    converted_tools: List[BaseTool] = []
    for server in mcp_tools:
        if server.mcp_servers is None or len(server.mcp_servers) <= 0:
            continue
        mcp_tools = server.tools
        try:
            converted_connections = await convert_connections(server.mcp_servers, server.hosted)
            for _, connection in converted_connections.items():
                if connection is None:
                    continue

                for mcp_tool in mcp_tools:
                    mcp_tool = MCPTool.model_validate(mcp_tool)
                    converted_tool = convert_mcp_tool_to_langchain_tool(None, mcp_tool, connection=connection)
                    converted_tools.append(converted_tool)
        except Exception as e:
            logger.exception(
                f"[_create_mcp_tools_async]-Failed to convert MCP tools for server {server.server_name} with error: {e}")
            continue

    if len(converted_tools) <= 0:
        return None
    return converted_tools


class DefaultChatAgentService(BaseChatAgentService):
    async def _setup_default_agent(self, chat_request: ChatRequestVO | ChatBatchRequestVO, **kwargs) -> \
            CompiledStateGraph[
                AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
            ]:
        used_model_config = chat_request.used_model_config
        retriever_config = chat_request.retriever_config
        self.retriever_config = retriever_config
        chat_session_upload_file_info = chat_request.chat_session_upload_file_info
        history_messages = chat_request.history_messages
        have_context = used_model_config.have_context
        self.is_chinese_text = is_chinese_text = used_model_config.is_chinese_text

        self.fall_back_answer = FALL_BACK_ANSWER
        if self.is_chinese_text:
            self.fall_back_answer = FALL_BACK_ANSWER_ZH

        tools: List[BaseTool] = []
        user_id = get_user_id()

        additional_llm_config = None
        if chat_session_upload_file_info:
            self.chat_session_state = ChatSessionState(
                uploaded_files_info={chat_session_upload_file_info.file_sync_record_id: chat_session_upload_file_info}
            )
            # 如果有上传文件信息，则添加相应的文件处理工具
            retriever_tool = create_search_relevant_documents_from_uploaded_files_tool(chat_session_upload_file_info)
            if retriever_tool is not None:
                tools.append(retriever_tool)
            # 使用工厂函数创建带有自定义描述的总结工具
            summarize_tool = create_summarize_chat_session_uploaded_file_tool(chat_session_upload_file_info)
            tools.append(summarize_tool)
            # 添加上传文件分页读取工具
            tools.append(read_uploaded_file_content_tool)
            tools.append(get_current_date_time_tool)
        elif have_context:  # 指的是知识库检索
            tools.append(search_relevant_documents_from_knowledge_base_tool)
            if can_use_get_build_task_files_detail_tool(user_id, retriever_config.knowledge_index_folder_id,
                                                        retriever_config.version):
                tools.append(get_knowledge_base_files_detail_tool)
            tools.append(summarize_knowledge_base_file_tool)
            tools.append(read_file_content_from_knowledge_tool)
            tools.append(get_current_date_time_tool)
        additional_llm_config = {"parallel_tool_calls": False}

        converted_mcp_tools = await create_mcp_tools_async(chat_request.used_mcp_tools)
        if converted_mcp_tools is not None and len(converted_mcp_tools) > 0:
            tools.extend(converted_mcp_tools)

        if self.react_system_prompt is None:
            self.react_system_prompt = self.langchain_llm.get_react_system_prompt(have_context=have_context,
                                                                                  is_chinese_text=is_chinese_text,
                                                                                  have_plugins=len(tools) > 0)
            react_prompt_str = self.react_system_prompt.format()
        else:
            if isinstance(self.react_system_prompt, str):
                react_prompt_str = self.react_system_prompt
            else:
                react_prompt_str = self.react_system_prompt.format()
        for message in history_messages:
            if message.role == "system":
                message.content = message.content + react_prompt_str

        # 计算上下文 token 限制
        self.context_tokens_limit = SupportedModelsInfoUtil.find_supported_model_by_id(
            chat_request.used_model_config.llm_id
        ).get("context_tokens_limit", DEFAULT_CONTEXT_TOKENS_LIMIT)
        model = self.langchain_llm.get_llm_client(additional_llm_config=additional_llm_config)

        # 更新 context 以包含 llm_client
        self.context = ApplicationContext(
            user_id=user_id,
            llm_client=model,
        )

        agent = create_agent(
            model,
            context_schema=ApplicationContext,
            state_schema=ChatState,
            tools=tools,
        )
        return agent

    def _clean_messages_for_unformatted_tool_messages(self, messages: list[ChatMessage]) -> list:
        """
        清理消息列表，移除所有 role 为 tool 的消息，
        以及带有 tool_calls 的 assistant 消息（避免孤立的 tool_calls）。
        """
        return [
            msg for msg in messages
            if msg.role != "tool" and not (msg.role == "assistant" and msg.tool_calls)
        ]

    def _check_agent_stopped_issue(self, chunk: dict):
        if output := chunk.get("output", None):
            if isinstance(output, str) and output.startswith("Agent stopped due to"):
                logger.warning(f"[_check_agent_stopped_issue]-Agent stopped with reason: {output}")
                return True
        return False

    def _prune_relevant_docs(self, retriever_docs: List[Document]) -> List[Document]:
        for index, doc in enumerate(retriever_docs):
            if doc.page_content is None or doc.page_content.strip() == "":
                logger.warning(f"[RetrieverDocsTool]- Document {doc.id} has empty page_content, removing it.")
                del retriever_docs[index]
                continue
            if doc.metadata is not None:
                doc.metadata = MetadataConstants.popup_useless_metadata(doc.metadata)
        return retriever_docs

    async def completions_stream(self,
                                 chat_request: ChatRequestVO,
                                 **kwargs):
        input = chat_request.query
        try:
            agent = await self._setup_default_agent(chat_request=chat_request)
        except Exception as e:
            logger.exception(f"[completions_stream]-Failed to init agent executor: {e}")
            raise e
        fall_back_resp_wrapper = ChatStreamResponseVO.create_content(content=self.fall_back_answer,
                                                                     resp_object="chat.completion.chunk")
        addition_info = {}

        try:
            history_messages = chat_request.history_messages
            history_messages = self._clean_messages_for_unformatted_tool_messages(history_messages)
            async for messages in agent.astream({
                "messages": prepare_input_messages(history_messages, input),
                "input": input,
                "retriever_config": self.retriever_config,
                "context_tokens_limit": self.context_tokens_limit,
                "chat_session_state": self.chat_session_state,
            },
                    config={
                        "recursion_limit": kwargs.get("agent_max_iterations", AGENT_MAX_ITERATIONS)
                    },
                    context=self.context,
                    stream_mode="messages"):
                # logger.info(f"[completions_stream_messages]-chunk:{messages}")

                if isinstance(messages, tuple):
                    for message in messages:
                        response = None
                        if isinstance(message, AIMessageChunk):
                            response = handle_ai_message_chunk(message, conversation_id=message.id)
                        elif isinstance(message, ToolMessage):
                            logger.debug(f"[completions_stream_messages]-ToolMessage:{message}")
                            response = handle_tool_message(message, addition_info)
                        elif message.__class__.__name__ == "ToolMessageChunk":
                            logger.debug(f"[completions_stream_messages]-ToolMessageChunk:{message}")
                            response = handle_tool_message_chunk(message, addition_info)
                        elif message.__class__.__name__ == "FunctionMessageChunk":
                            logger.debug(f"[completions_stream_messages]-FunctionMessageChunk:{message}")
                            response = handle_function_message_chunk(message)
                        else:
                            pass

                        if response:
                            yield response

        except GraphRecursionError as e:
            logger.error(f"[completions_stream]-Agent stopped due to max iterations with ex: {e}")
            yield "data: " + json.dumps(fall_back_resp_wrapper.dict(), ensure_ascii=False) + "\n\n"
        except RateLimitError as e:
            logger.error(f"[completions_stream]-Agent stopped due to rate limit with ex: {e}")
            self.fall_back_answer = RATE_LIMIT_ERROR_MESSAGE
            if self.is_chinese_text:
                self.fall_back_answer = RATE_LIMIT_ERROR_MESSAGE_ZH
            fall_back_resp_wrapper = ChatStreamResponseVO.create_content(content=self.fall_back_answer,
                                                                         resp_object="chat.completion.chunk")
            yield "data: " + json.dumps(fall_back_resp_wrapper.dict(), ensure_ascii=False) + "\n\n"
        except Exception as e:
            logger.exception(f"[completions_stream]-Failed to init agent executor: {e}")
            self.fall_back_answer = str(e)
            fall_back_resp_wrapper = ChatStreamResponseVO.create_content(content=self.fall_back_answer,
                                                                         resp_object="chat.completion.chunk")
            yield "data: " + json.dumps(fall_back_resp_wrapper.dict(), ensure_ascii=False) + "\n\n"

        if addition_info is not None and len(addition_info) > 0:
            addition_info_resp = ChatStreamResponseVO.create_addition_info(
                retriever_docs=addition_info.get("retriever_docs_info", None),
                search_keywords=addition_info.get("search_keywords", None))
            yield "data: " + json.dumps(addition_info_resp.dict(), ensure_ascii=False) + "\n\n"
        yield "data: [DONE]"

    async def completions(self,
                          chat_request: ChatRequestVO,
                          **kwargs) -> ChatResponseVO:
        input = chat_request.query
        output = self.fall_back_answer
        response_metadata = {}
        id = chat_request.id
        retriever_docs = []
        used_function_ids = []
        retriever_artifact = None

        try:
            agent = await self._setup_default_agent(chat_request=chat_request,
                                                    max_iterations=kwargs.get("agent_max_iterations",
                                                                              AGENT_MAX_ITERATIONS))

            results = await agent.ainvoke(
                {
                    "messages": prepare_input_messages(chat_request.history_messages, input),
                    "input": input,
                    "retriever_config": self.retriever_config,
                    "context_tokens_limit": self.context_tokens_limit,
                    "chat_session_state": self.chat_session_state,
                },
                config={
                    "recursion_limit": kwargs.get("agent_max_iterations", AGENT_MAX_ITERATIONS),
                },
                context=self.context
            )
            if results and (messages := results.get("messages", [])):
                for message in messages:
                    # 1\) 记录工具调用名称（用于 used_function_ids）
                    if isinstance(message, AIMessage):
                        if (tool_calls := message.tool_calls) is not None and len(tool_calls) > 0:
                            for tool_call in tool_calls:
                                if (function_name := tool_call.get("name")):
                                    used_function_ids.append(function_name)

                    # 2\) 捕获检索工具的 ToolMessage.artifact
                    if isinstance(message, ToolMessage) and message.name == SEARCH_RELEVANT_DOCS_TOOL_NAME:
                        artifact = getattr(message, "artifact", None)
                        if isinstance(artifact, dict):
                            retriever_artifact = artifact

                # 3\) 取最后一个 AIMessage 作为模型最终回答
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    output = last_message.content
                    response_metadata = last_message.response_metadata
                    id = last_message.id or id

            # 4\) 如果拿到了 artifact，就从中取出 retriever_docs_info
            if retriever_artifact is not None:
                docs_from_artifact = retriever_artifact.get("retriever_docs_info", []) or []
                # _prune_relevant_docs 已经负责清洗 metadata、剔除空内容
                retriever_docs = self._prune_relevant_docs(docs_from_artifact)
        except GraphRecursionError as e:
            logger.error(f"[completions]-Agent stopped due to max iterations: {e}")
            output = self.fall_back_answer
        except RateLimitError as e:
            logger.error(f"[completions]-Rate limit error: {e}")
            output = RATE_LIMIT_ERROR_MESSAGE_ZH if self.is_chinese_text else RATE_LIMIT_ERROR_MESSAGE
        except Exception as e:
            logger.exception(f"[completions]-Unexpected error: {e}")

        return ChatResponseVO(
            id=id,
            model=chat_request.used_model_config.llm_model_name,
            object="chat.completion",
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output),
                finish_reason=response_metadata.get("finish_reason", "stop")
            )],
            created=int(datetime.now().timestamp()),
            retriever_docs=retriever_docs or None,
            used_function_ids=used_function_ids or None
        )
