from typing import List, Optional

from langchain_core.documents import Document

from configs.model_config import logger
from controllers.common.retry_making_index_context import CreateIndexRetryableContext, retryable_step
from controllers.controller_util import add_created_date_on_doc_metadata
from controllers.dispatch_handlers.base_making_index_handler import BaseBuildingIndexHandler
from controllers.enums.create_index_steps_enum import CreateElasticSearchIndexStepsEnum
from core.common.common_util import documents_id_filling
from utils.elastic_search_client import ElasticSearchClient


class DefaultBuildingElasticSearchIndexHandler(BaseBuildingIndexHandler):
    def __init__(self,
                 is_chinese_text: bool = False, **kwargs):
        '''
            特别注意is_chinese_text参数的设置对ElasticSearch的中文分词以及搜索的召回率、准确率有影响，一定要提示让用户谨慎设置。
            :param kwargs: Pls refer to BaseUnstructuredFileLoader
        '''
        super().__init__(is_chinese_text=is_chinese_text, **kwargs)
        if self.user_folder_id:
            self.retryable_context = kwargs.get("retryable_context", CreateIndexRetryableContext(
                steps_enum=CreateElasticSearchIndexStepsEnum,
                current_step=CreateElasticSearchIndexStepsEnum.INITIATE.name,
                id=self.id,
                user_folder_id=self.user_folder_id))
        self.elastic_search_client = ElasticSearchClient(embedding_model=self.embedding_model)

    @retryable_step(steps_enum=CreateElasticSearchIndexStepsEnum, step=CreateElasticSearchIndexStepsEnum.CREATE_ES_SEARCH_INDEX.name)
    def _create_elastic_search_index(self,):
        index_name = self.user_folder_id + "ver" + str(self.checkout_version)
        self.elastic_search_client.create_index(index_name=index_name, is_chinese_text=self.is_chinese_text)
        logger.info(f"[create_index_and_save_to_storage]-Have created index: {index_name} in elastic search.")

    @retryable_step(CreateElasticSearchIndexStepsEnum, step=CreateElasticSearchIndexStepsEnum.LOAD_DOCS_AND_SAVE_TO_ES.name)
    def _load_docs_and_save_to_es(self, docs: Optional[List[Document]] = None, **kwargs) -> list[Document]:
        if docs is None:
            docs: List[Document] = self.load_docs(**kwargs)
            docs = documents_id_filling(documents=docs, document_id_prefix=self.id)

        docs = self.inject_metadata(docs, metadata=kwargs.get("metadata", {}))
        index_name = self.user_folder_id + "ver" + str(self.checkout_version)
        self.elastic_search_client.add_or_update_index_data(index_name=index_name, docs=docs)
        # docs = replace_words_on_doc_page_content(docs)
        self.docs_index_count = len(docs)
        logger.info(f"[create_index_and_save_to_storage]-Docs count: {self.docs_index_count}")
        self.retryable_context_info.update({"docs": docs})
        return docs

    def precompute_and_save_results(self,
                                    docs: Optional[List[Document]] = None,
                                    **kwargs):
        '''
            Use ParentDocumentRetriever add documents to local vector store

            :param kwargs:
                file_name: str
        '''
        self._create_elastic_search_index()

        docs = self._load_docs_and_save_to_es(docs=docs, **kwargs)
        self.retryable_request_params = {
            "docs": docs,
            **kwargs,
        }

        self._housekeeping_precompute_index_cache()
        return docs

    def del_part_of_indies(self, ids: List[str], **kwargs):
        assert self.user_folder_id is not None, "user_folder_id should not be None"
        assert self.checkout_version is not None, "checkout_version should not be None"
        index_name = self.user_folder_id + "ver" + str(self.checkout_version)
        self.elastic_search_client.delete_documents_in_index(index_name=index_name, doc_ids=ids)

    def del_version_of_indies_and_docs(self, **kwargs):
        """
        1. 删除elasticsearch index
        """
        index_name = self.user_folder_id + "ver" + str(self.checkout_version)
        self.elastic_search_client.delete_index(index_name=index_name)

    def del_workspace(self, **kwargs):
        all_indexes = self.elastic_search_client.find_indices_with_prefix(self.user_folder_id)
        for index in all_indexes:
            self.elastic_search_client.delete_index(index_name=index)
            logger.info(f"[del_workspace]-Have delete index: {index} from elastic search.")