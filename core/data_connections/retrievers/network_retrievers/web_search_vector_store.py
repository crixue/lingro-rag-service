import time
from typing import Optional, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from configs.model_config import logger, WEB_SEARCH_COUNT
from controllers.common.common_constant import MetadataConstants
from controllers.dispatch_handlers.making_vector_store_index_handler import ContentStringBuildingVectorStoreIndexHandler
from controllers.enums.source_type_enum import SourceTypeEnum
from core.data_connections.retrievers.local_vectorstore_retriever.custom_parent_document_retriever import ParentDocumentOfFAISSRetriever
from core.data_connections.retrievers.network_retrievers import SearchFetchParserServiceCall
from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import CustomLocalVectorDecorator
from utils.common_util import random_string, detect_if_is_chinese_text


class WebSearchVectorStoreDecorator:
    def __init__(self,
                 is_chinese_text: bool = False,
                 embedding_model: Embeddings = None,
                 embedding_model_name: str = None,
                 ignore_content_len: int = 120, **kwargs) -> None:
        self.is_chinese_text = is_chinese_text
        self.service_call = SearchFetchParserServiceCall()
        self.ignore_content_len = ignore_content_len
        self.vs_folder_name = 'network/tmp-' + random_string(8)
        self.local_vs_decorator = CustomLocalVectorDecorator(folder_name=self.vs_folder_name,
                                                             is_chinese_text=is_chinese_text,
                                                             embedding_model=embedding_model,
                                                             embedding_model_name=embedding_model_name,
                                                             )
        kwargs.pop('is_chinese_text', None)
        kwargs.pop('local_vs_folder_name', None)
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.making_index_handler = ContentStringBuildingVectorStoreIndexHandler(
            is_chinese_text=self.is_chinese_text,
            local_doc_folder_name=self.vs_folder_name,
            user_folder_id=self.vs_folder_name,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            **kwargs)
        self.kwargs = kwargs

    def web_search_docs(self, query: str, search_count: Optional[int] = WEB_SEARCH_COUNT) -> List[Document]:
        start_time = time.time()
        search_results = self.service_call.search_results(query, search_count)
        all_docs = []
        for item in search_results:
            is_chinese_text = True
            if item['text'] and len(item['text']) > 0:
                is_chinese_text = detect_if_is_chinese_text(item['text'])

            # 针对不同的文本设置不同的语言，为了后续的分词效果更好
            content_kg_index_handler = ContentStringBuildingVectorStoreIndexHandler(
                is_chinese_text=is_chinese_text,
                local_doc_folder_name=self.vs_folder_name,
                user_folder_id=self.vs_folder_name,
                embedding_model=self.embedding_model,
                embedding_model_name=self.embedding_model_name,
                **self.kwargs)
            docs: List[Document] = content_kg_index_handler.load_and_split_docs(
                content=item['text'],
                metadata={MetadataConstants.URL: item['url'], MetadataConstants.TITLE: item["title"], MetadataConstants.SOURCE_TYPE: SourceTypeEnum.WEB_SEARCH.value})
            all_docs.extend(docs)
        if len(all_docs) == 0:
            logger.warn(f"[web_search_docs]-No available paragraphs. The references provide no useful information.")
            all_docs = [Document(page_content="Not found any related info",
                                 metadata={"source_type": SourceTypeEnum.WEB_SEARCH.value})]
            return all_docs
        end_time = time.time()
        # 保留两位小数
        cost_time = round(end_time - start_time, 2)
        logger.info(f"[web_search_docs]- by query: {query} Search finished, time cost: {cost_time} seconds")
        return all_docs

    def add_documents(self,
                      docs: List[Document],
                      **kwargs):
        self.making_index_handler.precompute_and_save_results_multi_threads(docs, **self.kwargs)

    def as_retriever(self, **search_kwargs) -> ParentDocumentOfFAISSRetriever:
        parent_doc_retriever = self.local_vs_decorator.as_retriever(**{ **search_kwargs, "is_chinese_text": self.is_chinese_text})
        return parent_doc_retriever

    def remove_vector_store(self):
        self.local_vs_decorator.remove_all_vector_store()
