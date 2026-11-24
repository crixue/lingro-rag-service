import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from configs.model_config import *
from controllers.common.common_constant import DOC_STORE_SUFFIX
from core.common.common_constant import PLACEHOLDER_STRING
from core.data_connections.retrievers.local_vectorstore_retriever.custom_parent_document_retriever import ParentDocumentOfFAISSRetriever
from core.data_connections.retrievers.storage.documents_storage import BaseDocumentsStorage, DefaultDocumentsStorage
from core.data_connections.text_spitters.default_text_splitter import DefaultTextSplitter
from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2


class CustomLocalVectorDecorator:
    def __init__(self,
                 folder_name: str = None,
                 is_chinese_text: bool = True,
                 embedding_model: Optional[Embeddings] = None,
                 embedding_model_name: Optional[str] = None,
                 index_name: Optional[str] = "index",
                 faiss_instance: Optional[FAISS] = None,
                 **kwargs) -> None:
        """
        vs_path = VS_ROOT_PATH/embedding_model/folder_name
        Args:
            folder_name: self.vs_base_path / folder_name saves the faiss indies.
            e.g. self.vs_base_path = r"H:\\PycharmProjects\\langchain-ChatGLM\\vector_store\\all-mpnet-base-v2"
                            then folder_name = "tech_articles"
                the vs_path is "H:\PycharmProjects\langchain-ChatGLM\vector_store\all-mpnet-base-v2\tech_articles"
        """
        assert folder_name is not None, "folder_name should not be None"
        self.folder_name = folder_name
        self.is_chinese_text = is_chinese_text
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            self.embedding_model, self.embedding_model_name = DefaultEmbeddingsV2.get_static_embeddings(
                is_chinese_text=is_chinese_text, )
        self.vs_path = os.path.join(KB_ROOT_PATH, self.folder_name)
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.index_name = index_name
        self.faiss_instance = faiss_instance
        self.doc_store_path = os.path.join(KB_ROOT_PATH, self.folder_name + DOC_STORE_SUFFIX)
        if not os.path.exists(self.doc_store_path):
            os.makedirs(self.doc_store_path)
        self.document_storage: BaseDocumentsStorage = DefaultDocumentsStorage(root_path=self.doc_store_path)
        self.cn_chunk_size = kwargs.get("max_characters", CHARACTER_TEXT_SPLITTER['cn_chunk_size'])
        self.cn_chunk_overlap = kwargs.get("overlap", CHARACTER_TEXT_SPLITTER['cn_chunk_overlap'])
        self.en_chunk_size = kwargs.get("max_characters", CHARACTER_TEXT_SPLITTER['en_chunk_size'])
        self.en_chunk_overlap = kwargs.get("overlap", CHARACTER_TEXT_SPLITTER['en_chunk_overlap'])

        self.child_cn_chunk_size = kwargs.get("child_cn_chunk_size", CHARACTER_TEXT_SPLITTER['child_cn_chunk_size'])
        self.child_cn_chunk_overlap = kwargs.get("child_cn_chunk_overlap", CHARACTER_TEXT_SPLITTER['child_cn_chunk_overlap'])
        self.child_en_chunk_size = kwargs.get("child_en_chunk_size", CHARACTER_TEXT_SPLITTER['child_en_chunk_size'])
        self.child_en_chunk_overlap = kwargs.get("child_en_chunk_overlap", CHARACTER_TEXT_SPLITTER['child_en_chunk_overlap'])
        self.cn_separators = kwargs.get("cn_separators", DEFAULT_SEPARATOR['cn'])
        self.en_separators = kwargs.get("en_separators", DEFAULT_SEPARATOR['en'])

    @property
    def current_vs_path(self):
        return self.vs_path

    @property
    def embeddings(self):
        return self.embedding_model

    @property
    def docstore(self):
        return self.document_storage

    @property
    def child_splitter(self):
        return DefaultTextSplitter.get_text_splitter(
            splitter_type='recursive_character',
            chunk_size=self.child_cn_chunk_size if self.is_chinese_text else self.child_en_chunk_size,
            chunk_overlap=self.child_cn_chunk_overlap if self.is_chinese_text else self.child_en_chunk_overlap,
            separators=self.cn_separators if self.is_chinese_text else self.en_separators,
            keep_separator='end'
        )

    @property
    def parent_splitter(self):
        return DefaultTextSplitter.get_text_splitter(
            splitter_type='recursive_character',
            chunk_size=self.cn_chunk_size if self.is_chinese_text else self.en_chunk_size,
            chunk_overlap=self.cn_chunk_overlap if self.is_chinese_text else self.en_chunk_overlap,
            separators=self.cn_separators if self.is_chinese_text else self.en_separators,
            keep_separator='end'
        )

    def as_retriever(self, **kwargs) -> ParentDocumentOfFAISSRetriever:
        """
        search_kwargs (Optional[Dict]): Keyword arguments to pass to the
            search function. Can include things like:
            k: Amount of documents to return (Default: 4)
            score_threshold: Minimum relevance threshold
                    for similarity_score_threshold
            fetch_k: Amount of documents to pass to MMR algorithm
                    (Default: 20)
            lambda_mult: Diversity of results returned by MMR;
                    1 for minimum diversity and 0 for maximum. (Default: 0.5)
            filter: Filter by document metadata. Can use a filter to only retrieve documents from a specific paper
                docsearch.as_retriever(
                        search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
                    )
        """
        search_kwargs = {
            'top_k': kwargs.get('top_k', 4),
            'k': kwargs.get('k', None),
            'filter': kwargs.get('filter', None),
            '_normalize_L2': kwargs.get('normalize_L2', True),
            'fetch_k': kwargs.get('fetch_k', 20),
            'is_chinese_text': kwargs.get('is_chinese_text', self.is_chinese_text),
        }
        parent_doc_retriever = ParentDocumentOfFAISSRetriever(
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            vectorstore=self.local_vector_stores(),
            document_storage=self.document_storage,
            saved_index_folder_name=self.folder_name,
            embedding_model=self.embedding_model,
            search_kwargs=search_kwargs,
            **kwargs)
        return parent_doc_retriever

    def exists_local_faiss_indies(self) -> bool:
        """Check if the faiss indies exists.
        """
        if not os.path.exists(self.vs_path):
            return False
        path = Path(self.vs_path)
        exists_faiss = False
        exists_pkl = False
        for index_file in path.glob("*.faiss"):
            exists_faiss = True
        for pkl_file in path.glob("*.pkl"):
            exists_pkl = True
        return exists_faiss and exists_pkl

    def local_vector_stores(self, **kwargs) -> FAISS:
        if self.faiss_instance is not None:
            return self.faiss_instance
        if not self.exists_local_faiss_indies():
            if not os.path.exists(self.vs_path):
                os.makedirs(self.vs_path)

            # For now, there are two distance strategies: DistanceStrategy.EUCLIDEAN_DISTANCE and DistanceStrategy.MAX_INNER_PRODUCT.
            distance_strategy = kwargs.get("distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE)
            self.faiss_instance = FAISS.from_documents([Document(page_content=PLACEHOLDER_STRING)],
                                                       self.embedding_model,
                                                       normalize_L2=kwargs.get("normalize_L2", True),
                                                       distance_strategy=distance_strategy, )
            self.faiss_instance.save_local(folder_path=self.vs_path, index_name=self.index_name)
            return self.faiss_instance

        self.faiss_instance = FAISS.load_local(folder_path=self.vs_path,
                                               embeddings=self.embedding_model,
                                               index_name=self.index_name,
                                               allow_dangerous_deserialization=True)
        return self.faiss_instance

    def remove_all_vector_store(self):
        """Remove the vector store in the vs_path dir.
        """
        if self.vs_path is None or not os.path.exists(self.vs_path):
            return
        shutil.rmtree(self.vs_path)
        if self.document_storage:
            self.document_storage.delete_storage_documents()
        logger.info(f"Remove all vector store in {self.vs_path} and clear all documents")

    def remove_ids_in_index(self, ids: List[str], sub_ids: Optional[List[str]]):
        faiss_instance = self.local_vector_stores()
        if sub_ids is not None:
            has_deleted = faiss_instance.delete(sub_ids)
            logger.info(f"Delete {len(sub_ids)} sub_ids in index {self.vs_path}, has_deleted: {has_deleted}")
            faiss_instance.save_local(folder_path=self.vs_path, index_name=self.index_name)
        if self.document_storage:
            self.document_storage.delete_documents(ids)
            logger.info(f"Delete {len(ids)} docs in document_storage from: {self.doc_store_path}")
