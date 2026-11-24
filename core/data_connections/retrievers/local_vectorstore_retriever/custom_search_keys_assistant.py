from typing import List

from langchain.chains import LLMChain
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from configs.model_config import logger, DEFAULT_CONTEXT_TOKENS_LIMIT
from controllers.vo.startup_llm_model_request_vo import RequestMessage

DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["history_messages","question"],
    template="""You are an AI language model assistant. 
    Your task is to generate less or equal than 3 short questions to search from search engines. 
    It based on user original question and given history messages. 
    Each question should less than 10 words. Provide each question as a separate line.
    History messages: {history_messages}.
    Original question: {question}""",
)


def reduce_history_messages_tokens_below_limit(history_messages: List[RequestMessage], max_tokens_limit: int) -> List[RequestMessage]:
    num_messages = len(history_messages)
    if max_tokens_limit:
        tokens = [
            # self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
            len(message.content)
            for message in history_messages
        ]
        # Only keep the last of n messages that are below the token limit
        token_count = sum(tokens[-num_messages:])
        while token_count > max_tokens_limit:
            num_messages -= 1
            token_count -= tokens[-num_messages]
    return history_messages[-num_messages:]



def generate_new_query_keywords(question: str,
                                history_messages: List[RequestMessage],
                                llm: BaseChatModel,
                                prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
                                max_tokens_limit: int = DEFAULT_CONTEXT_TOKENS_LIMIT,
                                ) -> str:
    """Generate queries based upon user input.

    Args:
        question: user query

    Returns:
        List of LLM generated queries that are similar to the user input
    """
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=LineListOutputParser())
    new_question = ""
    history_messages = reduce_history_messages_tokens_below_limit(history_messages=history_messages, max_tokens_limit=max_tokens_limit)
    # contact history messages
    history_messages_concat = ""
    for message in history_messages:
        history_messages_concat += message.json(ensure_ascii=False) + ","
    try:
        line_list = llm_chain.run({"question": question, "history_messages": history_messages_concat})
        lines = line_list.lines
        if lines is not None:
            for line in lines:
                if line is not None:
                    # trim line str
                    line = line.strip()
                    new_question += " " + line
        return new_question
    except Exception as e:
        logger.error(f"Error in generate_keywords: {e}")
    return question