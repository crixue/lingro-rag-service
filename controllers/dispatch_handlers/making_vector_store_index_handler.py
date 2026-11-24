import copy
import copy
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from langchain_core.documents import Document

from configs.model_config import logger, KB_ROOT_PATH
from controllers.common.retry_making_index_context import retryable_step, CreateIndexRetryableContext
from controllers.controller_util import replace_words_on_doc_page_content
from controllers.dispatch_handlers.base_making_index_handler import BaseBuildingIndexHandler
from controllers.enums.create_index_steps_enum import CreateVectorStoreIndexStepsEnum
from core.data_connections.document_loaders.file_loader_v2 import ContentStringLoader
from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import CustomLocalVectorDecorator


class DefaultBuildingVectorStoreHandler(BaseBuildingIndexHandler):
    def __init__(self,
                 is_chinese_text: bool = False,
                 **kwargs):
        '''
            Making vector store for file types.
            :param kwargs: Pls refer to BaseUnstructuredFileLoader
        '''
        super().__init__(is_chinese_text=is_chinese_text, **kwargs)
        logger.info(f"[DefaultMakingVectorStoreHandler]-Ready for create faiss index from {self.local_doc_folder_name} for local vector store.")
        self.docs_index_count = 0
        self.child_docs_index_count = 0
        self.faiss_decorator = CustomLocalVectorDecorator(folder_name=self.local_doc_folder_name,
                                                          is_chinese_text=is_chinese_text, **kwargs)
        self.local_vector_stores = self.faiss_decorator.local_vector_stores(**kwargs)
        user_folder_id = self.user_folder_id if hasattr(self, "user_folder_id") else f"tmp-{self.id}"
        self.retryable_context = kwargs.get("retryable_context", CreateIndexRetryableContext(
                steps_enum=CreateVectorStoreIndexStepsEnum,
                current_step=CreateVectorStoreIndexStepsEnum.INITIATE.name,
                id=self.id,
                user_folder_id=user_folder_id))

    @retryable_step(steps_enum=CreateVectorStoreIndexStepsEnum,
                    step=CreateVectorStoreIndexStepsEnum.LOAD_AND_ADD_DOCS_TO_STORAGE.name)
    def _load_and_add_docs_to_storage(self, docs: Optional[List[Document]] = None, **kwargs):
        '''
            Use ParentDocumentRetriever add documents to local vector store

            :param need_clear_local_vs: if need clear local vector store first
            :param docs: if docs is None, will split docs from file_path first then load these docs
        '''
        import time
        start_time = time.time()

        if docs is None:
            logger.info(f"[_load_and_add_docs_to_storage]-Start loading docs from file...")
            load_start = time.time()
            docs: List[Document] = self.load_docs(**kwargs)
            logger.info(
                f"[_load_and_add_docs_to_storage]-Load docs completed in {time.time() - load_start:.2f}s, got {len(docs)} docs")

        docs = self.inject_metadata(docs, metadata=kwargs.get("metadata", {}))

        docs = replace_words_on_doc_page_content(docs)

        docs = [item for item in docs if item.page_content and item.page_content.strip() != ""]

        parent_doc_retriever = self.faiss_decorator.as_retriever(**{**kwargs, "is_chinese_text": self.is_chinese_text})

        full_docs, sub_docs = parent_doc_retriever.save_documents_to_storage(docs,
                                                                             document_id_prefix=self.document_id_prefix)

        filter_sub_start = time.time()
        sub_docs = [item for item in sub_docs if item.page_content and item.page_content.strip() != ""]

        doc_ids = [doc.id for doc in full_docs if doc.id]
        if doc_ids:
            self.docs_index_count = len(doc_ids)
        sub_doc_ids = [doc.id for doc in sub_docs if doc.id]
        if sub_doc_ids:
            self.child_docs_index_count = len(sub_doc_ids)
        self.retryable_context_info.update({"sub_docs": sub_docs})

        total_time = time.time() - start_time
        logger.info(
            f"[_load_and_add_docs_to_storage]-Loaded and added {self.docs_index_count} parent docs and {self.child_docs_index_count} child docs to storage. Total time: {total_time:.2f}s")
        return sub_docs

    @retryable_step(steps_enum=CreateVectorStoreIndexStepsEnum, step=CreateVectorStoreIndexStepsEnum.BUILD_FAISS_INDEX.name)
    def _build_faiss_index_by_sub_docs(self, sub_docs: List[Document], need_clear_local_vs: bool = False, **kwargs):
        if need_clear_local_vs and self.faiss_decorator.exists_local_faiss_indies():
            self.faiss_decorator.remove_all_vector_store()

        parent_doc_retriever = self.faiss_decorator.as_retriever(**{**kwargs, "is_chinese_text": self.is_chinese_text})
        parent_doc_retriever.build_faiss_index(sub_docs, vector_index_folder_path=self.faiss_decorator.current_vs_path)

    @retryable_step(steps_enum=CreateVectorStoreIndexStepsEnum, step=CreateVectorStoreIndexStepsEnum.BUILD_BM25_INDEX.name)
    def _build_bm25_index_by_sub_docs(self, sub_docs: List[Document], **kwargs):
        parent_doc_retriever = self.faiss_decorator.as_retriever(**{**kwargs, "is_chinese_text": self.is_chinese_text})
        parent_doc_retriever.build_bm25_index(sub_docs)

    def precompute_and_save_results(self, docs: Optional[List[Document]] = None, need_clear_local_vs: bool = False, **kwargs):
        logger.info(f"[precompute_and_save_results]-Start to create faiss index from {self.local_doc_folder_name} for local vector store.")
        sub_docs = self._load_and_add_docs_to_storage(docs, **kwargs)
        self._build_faiss_index_by_sub_docs(sub_docs=sub_docs, need_clear_local_vs=need_clear_local_vs, **kwargs)
        self._build_bm25_index_by_sub_docs(sub_docs=copy.deepcopy(sub_docs), **kwargs)
        self._housekeeping_precompute_index_cache()

    def precompute_and_save_results_multi_threads(self, docs: Optional[List[Document]] = None,
                                                  need_clear_local_vs: bool = False, **kwargs):
        sub_docs = self._load_and_add_docs_to_storage(docs, **kwargs)

        # 先在当前线程中创建包含当前上下文的可调用对象
        import contextvars
        current_context = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(lambda: current_context.run(
                lambda: self._build_faiss_index_by_sub_docs(sub_docs=sub_docs, need_clear_local_vs=need_clear_local_vs,
                                                            **kwargs)))
            executor.submit(lambda: self._build_bm25_index_by_sub_docs(sub_docs=copy.deepcopy(sub_docs), **kwargs))

        self._housekeeping_precompute_index_cache()

    def del_part_of_indies(self, ids: List[str], **kwargs):
        sub_ids = kwargs.get("sub_ids", None)
        self.faiss_decorator.remove_ids_in_index(ids, sub_ids)

    def get_docs_index_count(self):
        return self.docs_index_count, self.child_docs_index_count

    def del_version_of_indies_and_docs(self, **kwargs):
        self.faiss_decorator.remove_all_vector_store()

    def del_workspace(self, **kwargs):
        workspace_path = os.path.join(KB_ROOT_PATH, self.user_folder_id)
        if os.path.exists(workspace_path):
            shutil.rmtree(workspace_path)


class ContentStringBuildingVectorStoreIndexHandler(DefaultBuildingVectorStoreHandler):
    def __init__(self,
                 is_chinese_text: bool = False, **kwargs):
        '''
            Making vector store for content string.
            :param kwargs: Pls refer to BaseUnstructuredFileLoader
        '''

        super().__init__(is_chinese_text=is_chinese_text, **kwargs)
        self.is_chinese_text = is_chinese_text
        logger.info(f"[ContentStringMakingVSController]-Setup FileLoaderComponent with "
                    f"is_chinese_text:{is_chinese_text}, and other args:{kwargs}...")

    def load_and_split_docs(self, content: str = None, metadata: dict = None, **kwargs) -> List[Document]:
        self.fileLoader = ContentStringLoader(is_chinese_text=self.is_chinese_text, content=content,  metadata=metadata, **kwargs)
        return self.fileLoader.load_and_split()
