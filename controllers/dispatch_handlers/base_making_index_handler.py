import os
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from configs.model_config import logger
from controllers.common.common_constant import MetadataConstants
from controllers.common.retry_making_index_context import CreateIndexRetryableContext
from controllers.controller_util import inject_label_to_metadata
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2
from utils.common_util import get_beijing_current_time


class BaseBuildingIndexHandler(ABC):
    def __init__(self,
                 ingest_file_path: str = None,
                 local_doc_folder_name: str = None,
                 is_chinese_text: bool = True,
                 document_id_prefix: str = None,
                 user_folder_id: str = None,
                 mode: str = "single",
                 *,
                 id: str = None,
                 checkout_version: int = None,
                 **kwargs):
        """
        Base class for making index controller.
        Default use local file storage to store the documents. And default the file path: KB_ROOT_PATH/{embedding_model_name}/{local_doc_folder_name}_doc_store

        @param ingest_file_path: str, the file path contains multi types file to ingest
        """
        self.id = id
        self.checkout_version = checkout_version
        self.local_doc_folder_name = local_doc_folder_name
        self.is_chinese_text = is_chinese_text
        self.user_folder_id = user_folder_id

        if document_id_prefix is None:
            self.document_id_prefix = self.id
        else:
            self.document_id_prefix = document_id_prefix

        if kwargs.get("separators") is None:
            kwargs["separators"] = ["[SEP]", "\n\n", "\n", " "]

        if ingest_file_path is None or not os.path.exists(ingest_file_path):
            logger.warning(f"[BaseMakingIndexHandler]- file_path:{ingest_file_path} is None or not exists")
        else:
            logger.debug(
                f"[BaseMakingIndexHandler]-Setup FileLoaderComponent with file_path:{ingest_file_path}, "
                f"is_chinese_text:{is_chinese_text}, mode:{mode} and other args:{kwargs}...")
            self.fileLoader = DefaultFileLoaderComponent(file_path=ingest_file_path, is_chinese_text=is_chinese_text,
                                                         mode=mode, **kwargs)

        self.embedding_model, self.embedding_model_name = DefaultEmbeddingsV2.get_static_embeddings(
            is_chinese_text=is_chinese_text, **kwargs)
        if kwargs.get("embedding_model") is not None:
            self.embedding_model = kwargs.get("embedding_model")
            self.embedding_model_name = kwargs.get("embedding_model_name")

        self.docs_index_count = 0

        self.retryable_context: CreateIndexRetryableContext | None = kwargs.get("retryable_context", None)
        self.retryable_request_params = {}
        self.retryable_context_info = kwargs.get("retryable_context_info", {})

    def load_docs(self, **kwargs) -> List[Document]:
        return self.fileLoader.load(**kwargs)

    def inject_metadata(self, docs: List[Document], metadata: dict) -> List[Document] | None:
        """
        Inject metadata to the documents
        :param metadata: dict, the metadata to inject
        :return: List[Document]
        """
        if docs is None or len(docs) == 0:
            return docs
        if metadata is None:
            metadata = {}
        metadata.update({MetadataConstants.CREATED_AT: get_beijing_current_time().strftime("%Y-%m-%d %H:%M:%S")})
        if self.is_chinese_text:
            metadata.update({MetadataConstants.IS_CHINESE_TEXT: self.is_chinese_text})
        if self.checkout_version:
            metadata.update({MetadataConstants.CHECKOUT_VERSION: self.checkout_version})
        if self.user_folder_id:
            metadata.update({MetadataConstants.KNOWLEDGE_FOLDER_ID: self.user_folder_id})
        docs = inject_label_to_metadata(docs, metadata)
        return docs

    @abstractmethod
    def precompute_and_save_results(self, docs: Optional[List[Document]] = None, **kwargs):
        raise NotImplementedError

    def _housekeeping_precompute_index_cache(self):
        if self.retryable_context is not None:
            self.retryable_context.delete_context_info()

    @abstractmethod
    def del_part_of_indies(self, ids: List[str], **kwargs):
        raise NotImplementedError

    def get_docs_index_count(self) -> Tuple[int, int]:
        """
        Get the count of docs and child docs in the storage

        :return: Tuple[int, int]
        """
        return self.docs_index_count, 0

    @abstractmethod
    def del_version_of_indies_and_docs(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def del_workspace(self, **kwargs):
        raise NotImplementedError

    def release(self):
        pass