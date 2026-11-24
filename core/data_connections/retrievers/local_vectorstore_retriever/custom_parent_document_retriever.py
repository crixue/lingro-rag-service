from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Sequence

import time
import numpy as np
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import ByteStore
from langchain_text_splitters import TextSplitter
from pydantic import model_validator

from configs.model_config import logger, EMBEDDING_MINI_BATCH_SIZE_FOR_SMALLER_DOCS
from controllers.common.common_constant import MetadataConstants
from core.common.common_util import generate_document_ids, generate_sub_document_ids
from core.data_connections.retrievers.local_vectorstore_retriever.bm25_document_index import BM25DocumentIndex
from core.data_connections.retrievers.storage.documents_storage import BaseDocumentsStorage
from core.data_connections.retrievers.utils.retrieve_util import parse_query, PARSE_INFO_KEY_PHRASES


class ParentDocumentOfFAISSRetriever(MultiVectorRetriever):
    """Retrieve small chunks then retrieve their parent documents.

    When splitting documents for retrieval, there are often conflicting desires:

    1. You may want to have small documents, so that their embeddings can most
        accurately reflect their meaning. If too long, then the embeddings can
        lose meaning.
    2. You want to have long enough documents that the context of each chunk is
        retained.

    The ParentDocumentRetriever strikes that balance by splitting and storing
    small chunks of data. During retrieval, it first fetches the small chunks
    but then looks up the parent ids for those chunks and returns those larger
    documents.

    Note that "parent document" refers to the document that a small chunk
    originated from. This can either be the whole raw document OR a larger
    chunk.

    Examples:

        .. code-block:: python

            # Imports
            from langchain.vectorstores import Chroma
            from langchain.embeddings import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain.storage import InMemoryStore

            # This text splitter is used to create the parent documents
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            # This text splitter is used to create the child documents
            # It should create documents smaller than the parent
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
            # The storage layer for the parent documents
            store = InMemoryStore()

            # Initialize the retriever
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
    """

    byte_store: Optional[ByteStore] = None

    vectorstore: FAISS
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""

    child_splitter: TextSplitter
    """The text splitter to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    document_storage: BaseDocumentsStorage

    is_chinese_text: bool = False

    saved_index_folder_name: str = None

    embedding_model: Embeddings = None

    child_metadata_fields: Optional[Sequence[str]] = None
    """Metadata fields to leave in child documents. If None, leave all parent document 
        metadata.
    """

    alpha: float = 0.75
    """
    Factor to balance the BM25 and FAISS scores. The final score is calculated as:
    alpha * faiss_score + (1 - alpha) * bm25_score
    """

    @model_validator(mode='before')
    def root_validate(cls, values):
        if values.get("byte_store") is None:
            assert values.get("document_storage").byte_store is not None, "Byte store must be provided"
            values["byte_store"] = values.get("document_storage").byte_store
        return values

    def _split_docs_for_adding(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            document_id_prefix: Optional[str] = None,
    ) -> Tuple[List[Document], List[Document]]:
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        document_id_prefix, doc_ids = generate_document_ids(document_id_prefix, len(documents), ids)

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for sub_i, _doc in enumerate(sub_docs):
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            doc.id = _id
            doc.metadata["prefix_id"] = document_id_prefix
            doc.metadata[self.id_key] = doc.id
            full_docs.append(doc)
        _, sub_ids = generate_sub_document_ids(document_id_prefix, len(docs))
        for i, sub_doc in enumerate(docs):
            sub_doc.id = sub_ids[i]
        return docs, full_docs


    def build_bm25_index(self, sub_documents: List[Document]) -> BM25DocumentIndex:
        """
        Build the BM25 index from the documents.
        """
        assert self.saved_index_folder_name is not None, "saved_index_folder_name should not be None"
        index = BM25DocumentIndex(self.saved_index_folder_name, )
        sub_documents = [doc for doc in sub_documents if doc.page_content]
        index.create_or_update_index(self.is_chinese_text, sub_documents)
        return index


    def build_faiss_index(self, sub_documents: List[Document], vector_index_folder_path: str, **kwargs) -> FAISS:
        """
        Build the FAISS index from the documents.
        """
        assert vector_index_folder_path is not None, "vector_index_folder_path must be provided"
        start_time = time.time()

        mini_batch = kwargs.get("mini_batch", EMBEDDING_MINI_BATCH_SIZE_FOR_SMALLER_DOCS)
        sub_documents = [doc for doc in sub_documents if doc.page_content and doc.page_content.strip() != ""]
        sub_ids = [doc.id for doc in sub_documents]
        for i in range(0, len(sub_documents), mini_batch):
            if isinstance(self.vectorstore, FAISS):
                self.vectorstore.add_documents(documents=sub_documents[i:i + mini_batch], ids=sub_ids[i:i + mini_batch])
        # Must save the faiss index after add documents!
        if isinstance(self.vectorstore, FAISS):
            self.vectorstore.save_local(folder_path=vector_index_folder_path, )
        end_time = time.time()
        logger.info(f"[build_faiss_index]-Build FAISS index with {len(sub_documents)} sub_documents in {end_time - start_time:.2f}s")
        return self.vectorstore

    def save_documents_to_storage(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            document_id_prefix: Optional[str] = None,
            add_to_docstore: bool = True,
    ) -> tuple[list[Document | None], list[Document | None]]:
        """
        1. Create index on vectorstores and BM25 index on the documents.
        2. Adds documents to the docstore.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        import time
        start_time = time.time()

        docs, full_docs = self._split_docs_for_adding(documents, ids, document_id_prefix)

        # Add documents to docstore
        if add_to_docstore and self.document_storage is not None:
            docstore_start = time.time()
            self.document_storage.save_documents(documents=full_docs, require_document_id_filling=False)
            logger.debug(
                f"[save_documents_to_storage]-Saving {len(full_docs)} documents to docstore completed in {time.time() - docstore_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"[save_documents_to_storage]-Total time: {total_time:.2f}s")
        return full_docs, docs

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Parsed query is a list of phrases. Please check the detail to refer prompt_definitions/builtin/lang/en/tool_definition_prompt.json
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        parsed_query_info = parse_query(query)
        phrases = parsed_query_info.get(PARSE_INFO_KEY_PHRASES, [])
        if len(phrases) > 0:
            query = " ".join(phrases)

        total_docs_with_scores = {}  # key: Document id; value: Document with score
        top_k = self.search_kwargs.get("top_k", 4)
        fetch_k = self.search_kwargs.get('fetch_k', 20)
        search_kwargs = {**self.search_kwargs, "k": min(top_k * 5, 20), "fetch_k": min(fetch_k * 5, 100)}
        def search_faiss():
            """
            从 local_vector_stores 的 FAISS 索引使用的度量（metric） distance_strategy 默认采用的是DistanceStrategy.EUCLIDEAN_DISTANCE进行相似度检索。
            所以这里的 score 越小表示越相似。
            如果更换成 DistanceStrategy.MAX_INNER_PRODUCT，则 score 越大表示越相似。
            """
            # Search from FAISS
            logger.info(f"[ParentDocumentRetriever] - Start searching in vectorstore for query: {query} with search_kwargs: {search_kwargs}")
            vector = self.embedding_model.embed_query(query)
            if self.search_type == SearchType.mmr:
                faiss_results: list[
                    tuple[Document, float]] = self.vectorstore.max_marginal_relevance_search_with_score_by_vector(
                    vector, **search_kwargs)
            else:
                faiss_results: list[tuple[Document, float]] = self.vectorstore.similarity_search_with_score_by_vector(
                    vector, **search_kwargs)
            for doc, score in faiss_results[:8]:
                logger.debug(f"[ParentDocumentRetriever] - FAISS retrieved doc: {doc} with score: {score}")

            return faiss_results

        def search_bm25():
            # Search from BM25
            bm25_results: list[tuple[Document, float]] = BM25DocumentIndex(
                self.saved_index_folder_name, ).similarity_search_with_score(
                query,
                self.search_kwargs.get("is_chinese_text", self.is_chinese_text),
                k=min(top_k * 5, 20)
            )
            return bm25_results

        import contextvars
        current_context = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 每个任务在提交前就复制当前上下文
            faiss_future = executor.submit(lambda: current_context.run(search_faiss))
            bm25_future = executor.submit(lambda: search_bm25())

            faiss_results = faiss_future.result()
            bm25_results = bm25_future.result()

        eps = 1e-6
        faiss_scores = np.array([score for _, score in faiss_results if score > 0])
        if len(faiss_scores) > 0:
            # 根据距离策略选择归一化方式
            if self.vectorstore.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
                # 对于欧几里得距离,距离越小越相似,需要反转
                normalized_faiss_scores = (max(faiss_scores) - faiss_scores) / (
                            max(faiss_scores) - min(faiss_scores) + eps)
            else:
                # 对于其他策略(如MAX_INNER_PRODUCT),分数越大越相似,保持原逻辑
                normalized_faiss_scores = (faiss_scores - min(faiss_scores)) / (
                            max(faiss_scores) - min(faiss_scores) + eps)
            for result, normalized_score in zip(faiss_results, normalized_faiss_scores):
                score = normalized_score * self.alpha
                doc: Document = result[0]
                if not doc:
                    continue

                doc_id = doc.metadata.get("doc_id")
                if doc_id is None:
                    logger.warning(f"[ParentDocumentRetriever] - Document id is None for document: {doc}")
                    continue
                if doc_id in total_docs_with_scores.keys():
                    last_doc = total_docs_with_scores[doc_id]
                    last_score = last_doc.metadata.get("score", 0.0)
                    score += last_score
                doc.metadata["score"] = score
                doc.metadata["faiss_score"] = score
                total_docs_with_scores[doc_id] = doc


        bm25_scores = np.array([score for _, score in bm25_results if score > 0])
        if len(bm25_scores) > 0:
            normalized_bm25_scores = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores)  + eps)
            for result, normalized_score in zip(bm25_results, normalized_bm25_scores):
                score = normalized_score * (1 - self.alpha)
                doc: Document = result[0]
                if not doc:
                    continue

                doc_id = doc.metadata.get("doc_id")
                if doc_id is None:
                    logger.warning(f"[ParentDocumentRetriever] - Document id is None for document: {doc}")
                    continue
                if doc_id in total_docs_with_scores.keys():
                    last_doc = total_docs_with_scores[doc_id]
                    last_score = last_doc.metadata.get("score", 0.0)
                    last_bm25_score = last_doc.metadata.get("bm25_score", 0.0)
                    last_doc.metadata["score"] = score + last_score
                    last_doc.metadata["bm25_score"] = score + last_bm25_score
                    doc = last_doc
                else:
                    doc.metadata["score"] = score
                    doc.metadata["bm25_score"] = score

                total_docs_with_scores[doc_id] = doc

        if len(total_docs_with_scores) == 0:
            logger.warning(f"[ParentDocumentRetriever] - No documents found for query: {query}")
            return []
        # reorder the results by the total score and pick top k
        top_k_docs_with_scores = sorted(total_docs_with_scores.items(), key=lambda x: x[1].metadata["score"], reverse=True)[:top_k]
        logger.info(f"[ParentDocumentRetriever] - Found {len(top_k_docs_with_scores)} documents from vectorstore for query: {query}")

        # We do this to maintain the order of the ids that are returned
        ids = []
        for doc_id, doc in top_k_docs_with_scores:
            ids.append(doc_id)
        docs = self.document_storage.retriever_documents(ids)
        for doc in docs:
            doc.metadata[MetadataConstants.BM25_SCORE] = total_docs_with_scores[doc.id].metadata.get("bm25_score", 0.0)
            doc.metadata[MetadataConstants.FAISS_SCORE] = total_docs_with_scores[doc.id].metadata.get("faiss_score", 0.0)
            doc.metadata[MetadataConstants.SCORE] = total_docs_with_scores[doc.id].metadata.get("score", 0.0)

        return docs
