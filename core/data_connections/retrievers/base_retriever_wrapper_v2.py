import time
from abc import ABC
from abc import abstractmethod
from typing import List

from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2


class BaseRetrieverWrapperV2(ABC):
    def __init__(self,
                 query_str: str = None,
                 llm: BaseChatModel or None = None,
                 is_chinese_text: bool = False,
                 retriever_type: BuildKnowledgeIndexTypeEnum = BuildKnowledgeIndexTypeEnum.WEB_SEARCH,
                 knowledge_index_folder_id: str = None,
                 knowledge_index_folder_name: str = None,
                 search_type: SearchType = SearchType.similarity,
                 search_top_k: int = 4,
                 use_embedding_filter_assist: bool = False,
                 use_reorder_assist: bool = False,
                 web_search_results_count: int = 8,
                 version: str = None,
                 embedding_model_id: str = None,
                 **kwargs) -> None:
        '''
        :param query_str: the query string to be used for retrieval
        :param llm: the LLM to be used for query generation
        :param is_chinese_text: whether the text is Chinese
        :param retriever_type: the type of retriever to be used. Must be one of "local_vector_stores", "web_search", or "fixed".
                And fixed means a combination of local_vector_stores and web_search
        :param knowledge_index_folder_name: the folder name of the local vector store to be used
        :param search_type: the type of search to be used. Must be one of "mmr" or "similarity"
        :param search_top_k: the number of documents to be retrieved
        :param use_embedding_filter_assist: whether to use embedding filter assist
        :param use_reorder_assist: whether to use reorder assist
        :param kwargs: other keyword arguments: fixed_weights, search_tool, ignore_content_len, is_jupyter_notebook, embedding_similarity_threshold.
                And fixed_weights is a list of two floats that sum to 1, means the weights of web_search: local_vector_stores .
        '''
        start_time = time.time()
        self.is_chinese_text = is_chinese_text
        self.retriever_type = retriever_type
        self.use_reorder_assist = use_reorder_assist
        self.query_str = query_str
        self.knowledge_index_folder_id = knowledge_index_folder_id
        self.knowledge_index_folder_name = knowledge_index_folder_name
        self.version = version
        self.search_top_k = search_top_k
        self.search_type = search_type
        self.default_retriever = None
        self.network_retriever = None
        self.multi_queries: List[str] = None
        self.use_embedding_filter_assist = use_embedding_filter_assist
        self.llm = llm

        kwargs["web_search_results"] = web_search_results_count
        if embedding_model_id is not None:
            embedding_model, embedding_model_name = DefaultEmbeddingsV2.get_static_embeddings(
                embedding_model_id=embedding_model_id,
                is_chinese_text=is_chinese_text)
        else:
            embedding_model, embedding_model_name = DefaultEmbeddingsV2.get_static_embeddings(
                is_chinese_text=is_chinese_text)
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.kwargs = kwargs

    @abstractmethod
    def get_retriever(self, **kwargs) -> BaseRetriever:
        raise NotImplementedError

    def get_multi_queries(self):
        return None

    def housekeeping(self):
        pass
