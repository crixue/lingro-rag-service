import json
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, ToolMessage

from configs.model_config import logger
from controllers.vo.startup_llm_model_request_vo import ChatStreamResponseVO

# 全局字典,用于累积 tool call chunks (按 conversation_id 或 request_id 隔离)
_tool_call_accumulators = {}


def _get_tool_call_accumulator(conversation_id: str = "default"):
    """获取或创建 tool call 累积器"""
    if conversation_id not in _tool_call_accumulators:
        _tool_call_accumulators[conversation_id] = {}
    return _tool_call_accumulators[conversation_id]


def _clear_tool_call_accumulator(conversation_id: str = "default"):
    """清除 tool call 累积器"""
    if conversation_id in _tool_call_accumulators:
        del _tool_call_accumulators[conversation_id]


def handle_ai_message_chunk(message: AIMessageChunk, conversation_id: str = "default") -> Optional[str]:
    """处理 AI 消息块,返回 SSE 格式的响应字符串

    Args:
        message: AIMessageChunk 消息对象
        conversation_id: 会话ID,用于隔离不同会话的 tool call 累积

    Returns:
        SSE 格式的字符串,如果无需输出则返回 None
    """
    content = message.text

    # 1. 处理工具调用 - 累积 chunks 并在完整时发送
    if message.tool_call_chunks is not None and len(message.tool_call_chunks) >= 1:
        message_tool_call_chunk = message.tool_call_chunks[-1]
        logger.info(f"[handle_ai_message_chunk]-tool call chunk:{message_tool_call_chunk}")

        if message_tool_call_chunk is not None:
            accumulator = _get_tool_call_accumulator(conversation_id)
            chunk_index = message_tool_call_chunk.get("index", 0)

            # 初始化或获取当前 index 的累积器
            if chunk_index not in accumulator:
                accumulator[chunk_index] = {
                    "name": None,
                    "args": "",
                    "id": None,
                    "index": chunk_index
                }

            current_call = accumulator[chunk_index]

            # 累积各个字段
            if message_tool_call_chunk.get("name") is not None:
                current_call["name"] = message_tool_call_chunk["name"]

            if message_tool_call_chunk.get("args") is not None:
                current_call["args"] += message_tool_call_chunk["args"]

            if message_tool_call_chunk.get("id"):
                current_call["id"] = message_tool_call_chunk["id"]

            # 检查是否完整 (简单判断: args 以 } 或 }" 结尾,且有 name)
            args_str = current_call["args"].strip()
            is_complete = (
                current_call["name"] is not None and
                args_str and
                (args_str.endswith('}') or args_str.endswith('}"'))
            )

            if is_complete:
                # 构造完整的 tool call 并发送
                try:
                    complete_tool_call = {
                        "name": current_call["name"],
                        "args": current_call["args"],
                        "id": current_call["id"],
                        "index": current_call["index"],
                        "type": "tool_call",
                        "status": "calling"
                    }

                    # 尝试解析 args 以验证完整性
                    try:
                        json.loads(current_call["args"])
                    except json.JSONDecodeError:
                        logger.warning(f"[handle_ai_message_chunk]-Tool call args not valid JSON yet: {current_call['args']}")
                        return None

                    resp_wrapper = ChatStreamResponseVO.create_content(
                        content="",
                        tool_call=complete_tool_call,
                        id=message.id,
                        resp_object="chat.completion.chunk",
                        content_type="tool_call")

                    # 清除已发送的累积器
                    del accumulator[chunk_index]

                    return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"

                except Exception as e:
                    logger.warning(f"[handle_ai_message_chunk]-Failed to serialize complete ToolCall: {e}")
                    return None
            else:
                # 尚未完整,不发送
                return None
        else:
            return None
    
    # 2. 处理 reasoning 内容块（新格式：在 message.content 列表中）
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get('type') == 'reasoning':
                # 从 reasoning 块中提取 summary
                summary_list = block.get('summary', [])
                if isinstance(summary_list, list) and len(summary_list) > 0:
                    # 收集所有 summary_text 类型的文本片段
                    summary_texts = []
                    for summary_item in summary_list:
                        if isinstance(summary_item, dict):
                            if summary_item.get('type') == 'summary_text':
                                text = summary_item.get('text', '')
                                if text:  # 允许空格等字符
                                    summary_texts.append(text)

                    # 如果有文本内容，则输出
                    if summary_texts:
                        combined_summary = ''.join(summary_texts)
                        logger.debug(
                            f"[handle_ai_message_chunk]-Found reasoning summary: {combined_summary[:100]}..."
                        )
                        resp_wrapper = ChatStreamResponseVO.create_content(
                            id=block.get('id', message.id),
                            content=combined_summary,
                            resp_object="chat.completion.chunk",
                            content_type="reasoning_content")
                        return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"

    # 3. 处理 additional_kwargs 中的 reasoning_content（旧格式兼容）
    additional_kwargs = message.additional_kwargs
    if additional_kwargs is not None:
        reasoning_content = additional_kwargs.get('reasoning_content', '')
        if reasoning_content and reasoning_content.strip() != "":
            logger.debug(
                f"[handle_ai_message_chunk]-Found reasoning_content: {reasoning_content[:100]}..."
            )
            resp_wrapper = ChatStreamResponseVO.create_content(
                id=message.id,
                content=reasoning_content,
                resp_object="chat.completion.chunk",
                content_type="reasoning_content")
            return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"

    # 4. 处理普通文本内容
    if content is None or content == "":
        return None

    resp_wrapper = ChatStreamResponseVO.create_content(
        id=message.id,
        content=content,
        resp_object="chat.completion.chunk")
    return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"


def serialize_tool_message_value(value):
    """将值转换为可序列化的格式,处理 Document 对象"""
    if isinstance(value, Document):
        return value.page_content
    elif isinstance(value, list):
        return [serialize_tool_message_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_tool_message_value(v) for k, v in value.items()}
    return value


def handle_tool_message(message: ToolMessage, addition_info: dict) -> Optional[str]:
    """处理工具消息，返回 SSE 格式的响应字符串

    Args:
        message: ToolMessage 消息对象
        addition_info: 用于存储额外信息的字典（会被修改）

    Returns:
        SSE 格式的字符串，如果无需输出则返回 None
    """
    # 收集检索工具的附加信息
    if (message.name is not None and
            (message.name in [
                'search_relevant_documents_from_uploaded_files_tool',
                'search_relevant_documents_from_knowledge_base_tool'
            ])):
        logger.debug(f"[handle_tool_message]-ToolMessage:{message}")
        if (artifact := message.artifact) is not None:
            addition_info.update({**artifact})
    else:
        logger.debug(f"[handle_tool_message]-ToolMessage:{message}")

    # 构造工具消息响应
    try:
        tool_message_info = {
            k: serialize_tool_message_value(v) for k, v in message.__dict__.items() if v is not None}
        if tool_message_info.get("tool_call_id", None) is not None:
            tool_message_info["id"] = tool_message_info["tool_call_id"]
        # content = json.dumps(tool_message_info, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[handle_tool_message]-Failed to serialize ToolMessage: {e}")
        return None

    resp_wrapper = ChatStreamResponseVO.create_content(
        content="",
        tool_call=tool_message_info,
        id=message.id,
        resp_object="chat.completion.chunk",
        content_type="tool_call")

    return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"


def handle_tool_message_chunk(message, addition_info: dict) -> Optional[str]:
    """处理工具消息块（ToolMessageChunk），返回 SSE 格式的响应字符串

    Args:
        message: ToolMessageChunk 消息对象
        addition_info: 用于存储额外信息的字典（会被修改）

    Returns:
        SSE 格式的字符串，如果无需输出则返回 None
    """
    # 收集检索工具的附加信息
    if (hasattr(message, 'name') and message.name is not None and
            message.name in [
                'search_relevant_documents_from_uploaded_files_tool',
                'search_relevant_documents_from_knowledge_base_tool'
            ]):
        logger.debug(f"[handle_tool_message_chunk]-ToolMessageChunk:{message}")
        if hasattr(message, 'artifact') and (artifact := message.artifact) is not None:
            addition_info.update({**artifact})
    else:
        logger.debug(f"[handle_tool_message_chunk]-ToolMessageChunk:{message}")

    # 构造工具消息块响应
    try:
        content = json.dumps({
            k: serialize_tool_message_value(v) for k, v in message.__dict__.items() if v is not None
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[handle_tool_message_chunk]-Failed to serialize ToolMessageChunk: {e}")
        return None

    resp_wrapper = ChatStreamResponseVO.create_content(
        content=content,
        id=getattr(message, 'id', None),
        resp_object="chat.completion.chunk",
        content_type="tool_call")

    return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"


def handle_function_message_chunk(message) -> Optional[str]:
    """处理函数消息块（FunctionMessageChunk），返回 SSE 格式的响应字符串

    Args:
        message: FunctionMessageChunk 消息对象

    Returns:
        SSE 格式的字符串，如果无需输出则返回 None
    """
    logger.debug(f"[handle_function_message_chunk]-FunctionMessageChunk:{message}")

    # 构造函数消息块响应
    try:
        content = json.dumps({
            k: serialize_tool_message_value(v) for k, v in message.__dict__.items() if v is not None
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[handle_function_message_chunk]-Failed to serialize FunctionMessageChunk: {e}")
        return None

    resp_wrapper = ChatStreamResponseVO.create_content(
        content=content,
        id=getattr(message, 'id', None),
        resp_object="chat.completion.chunk",
        content_type="tool_call")

    return "data: " + json.dumps(resp_wrapper.dict(), ensure_ascii=False) + "\n\n"

