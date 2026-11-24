import os
import os
import random
import threading
import time
from abc import ABC
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from configs.model_config import logger, NEO4J_CONFIG, KB_ROOT_PATH, WEB_SEARCH_EMBEDDING_MODELS_WEIGHTS, \
    WEB_SEARCH_COUNT
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from core.data_connections.retrievers.base_retriever_wrapper_v2 import BaseRetrieverWrapperV2
from core.data_connections.retrievers.elasticsearch_retriever.elasticsearch_retriever import \
    document_field_mapper, DefaultElasticsearchRetriever
from core.data_connections.retrievers.knowlege_graph_retrievers.hippo_retriever import HippoRetriever
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import GRAPH_DB_PREFIX
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.hipporag_with_graph_db import HippoRAG
from core.data_connections.retrievers.local_vectorstore_retriever.custom_parent_document_retriever import \
    ParentDocumentOfFAISSRetriever
from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import CustomLocalVectorDecorator
from core.data_connections.retrievers.mixin_retreiver_wrapper_v2 import MixedLocalVSAndWebSearchRetrieverWrapperV2, \
    MixedElasticsearchAndWebSearchRetrieverWrapperV2, MixedHippoKnowledgeGraphAndWebSearchRetrieverWrapperV2
from core.data_connections.retrievers.network_retrievers.web_search_vector_store import WebSearchVectorStoreDecorator
from core.data_connections.retrievers.storage.documents_storage import DefaultDocumentsStorage
from utils.elastic_search_client import ElasticSearchClient
from utils.neo4j_client import Neo4jGraphDBClient


class ElasticSearchRetrieverWrapperV2(BaseRetrieverWrapperV2, ABC):
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert self.knowledge_index_folder_id is not None, "knowledge_index_folder_id must be provided"
        assert self.version is not None, "version must be provided"
        top_k = self.search_top_k if self.search_top_k is not None else kwargs.get("top_k", 5)
        self.default_retriever = DefaultElasticsearchRetriever(
            es_client=ElasticSearchClient(embedding_model=self.embedding_model).es_client,
            index_name=self.knowledge_index_folder_id + "ver" + str(self.version),
            embedding_model=self.embedding_model,
            document_mapper=document_field_mapper,
            query_kwargs={"top_k": top_k,
                          "num_candidates": kwargs.get("num_candidates", 50)},
        )

    def get_multi_queries(self):
        return self.multi_queries

    def get_retriever(self, **kwargs):
        return self.default_retriever


class HippoRAGRetrieverWrapperV2(BaseRetrieverWrapperV2, ABC):
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.knowledge_index_folder_id is not None, "knowledge_index_folder_id must be provided"
        assert self.version is not None, "version must be provided"

        top_k = self.search_top_k if self.search_top_k is not None else kwargs.get("top_k", 4)
        dataset = kwargs.get("file_name", "default")
        self.neo4j_client = kwargs.get("neo4j_client", None) or Neo4jGraphDBClient(
            uri=kwargs.get("neo4j_uri", NEO4J_CONFIG["NEO4J_URI"]),
            user=kwargs.get("neo4j_user", NEO4J_CONFIG["NEO4J_USER"]),
            password=kwargs.get("neo4j_password", NEO4J_CONFIG["NEO4J_PASSWORD"]),
        )
        graph_db_name = GRAPH_DB_PREFIX + self.knowledge_index_folder_id + "ver" + str(self.version)
        save_path = os.path.join(KB_ROOT_PATH, f"{self.knowledge_index_folder_name}_doc_store")
        self.storage = DefaultDocumentsStorage(root_path=save_path)
        self.hippo_rag = HippoRAG(
            corpus_name=dataset,
            llm_client=self.llm,
            retriever_name=self.embedding_model_name,
            force_retrieve=kwargs.get("force_retrieve", False),
            local_doc_folder_name=self.knowledge_index_folder_name,
            checkout_version=self.version,
            graph_db_name=graph_db_name,
            neo4j_client=self.neo4j_client,
            embedding_model=self.embedding_model,
            doc_ensemble=True, dpr_only=False,
            user_folder_id=self.knowledge_index_folder_id,
            storage=self.storage,
        )
        self.default_retriever = HippoRetriever(hippo_rag=self.hippo_rag,
                                                docstore=self.storage,
                                                search_kwargs={"top_k": top_k},
                                                score_threshold=kwargs.get("score_threshold", 0.09),
                                                enable_logging=False
                                                )
        self.enable_logging = kwargs.get("enable_logging", False)

    def get_multi_queries(self):
        return self.multi_queries

    def get_retriever(self, **kwargs):
        return self.default_retriever


class LocalVectorStoreRetrieverWrapperV2(BaseRetrieverWrapperV2, ABC):
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.knowledge_index_folder_name is not None, "local_vs_folder_name must be provided"
        kwargs.pop("is_chinese_text", None)
        kwargs.pop("embedding_model", None)
        kwargs.pop("embedding_model_name", None)
        kwargs.pop("folder_name", None)
        vector_store_decorator = CustomLocalVectorDecorator(folder_name=self.knowledge_index_folder_name,
                                                            is_chinese_text=self.is_chinese_text,
                                                            embedding_model=self.embedding_model,
                                                            embedding_model_name=self.embedding_model_name,
                                                            **kwargs)
        if not vector_store_decorator.exists_local_faiss_indies():
            logger.error(
                f"[DefaultRetriever]-local_vs_path:{vector_store_decorator.current_vs_path} faiss index not exists!")
            raise ValueError(f"local_vs_path:{vector_store_decorator.current_vs_path} faiss index not exists!")
        self.default_retriever: ParentDocumentOfFAISSRetriever = vector_store_decorator.as_retriever(
            is_chinese_text=self.is_chinese_text,
            search_type=self.search_type,
            top_k=self.search_top_k,
            k=self.search_top_k)

        if self.use_embedding_filter_assist:
            from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
            embedding_similarity_threshold = kwargs.get("embedding_similarity_threshold", 0.5)
            relevant_filter = EmbeddingsFilter(embeddings=self.embedding_model,
                                               similarity_threshold=embedding_similarity_threshold)
            from langchain_community.document_transformers import EmbeddingsRedundantFilter
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_model)
            from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
            pipeline_compressor = DocumentCompressorPipeline(
                # transformers=[relevant_filter, redundant_filter]
                transformers=[relevant_filter]
            )
            from langchain_classic.retrievers import ContextualCompressionRetriever
            self.default_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                    base_retriever=self.default_retriever)
            logger.info(
                "[DefaultRetriever]-Using embedding filter assist and embedding_similarity_threshold: {}".format(
                    embedding_similarity_threshold))

    def get_multi_queries(self):
        return self.multi_queries

    def get_retriever(self):
        return self.default_retriever


class WebSearchVectorStoreRetrieverWrapperV2(BaseRetrieverWrapperV2, ABC):
    def _select_embedding_model_by_weight(self, weights_dict: Optional[dict] = None):
        """根据权重比例随机选择一个embedding模型

        Args:
            weights_dict (dict): 权重字典，默认使用WEB_SEARCH_EMBEDDING_MODELS_WEIGHTS

        Returns:
            str: 被选中的模型key
        """
        if weights_dict is None:
            weights_dict = WEB_SEARCH_EMBEDDING_MODELS_WEIGHTS

        # 获取所有key和对应的权重
        keys = list(weights_dict.keys())
        weights = list(weights_dict.values())

        # 使用random.choices进行加权随机选择
        selected_key = random.choices(keys, weights=weights, k=1)[0]

        return selected_key

    def __init__(self,
                 **kwargs) -> None:
        start_time = time.time()
        kwargs["embedding_model_id"] = self._select_embedding_model_by_weight(kwargs.get("weights_dict"))  # 设置web search的embedding model
        super().__init__(**kwargs)
        assert self.knowledge_index_folder_name is not None, "local_vs_folder_name must be provided"
        kwargs.pop("is_chinese_text", None)
        kwargs.pop("embedding_model", None)
        kwargs.pop("embedding_model_name", None)
        web_vector_store_decorator = WebSearchVectorStoreDecorator(is_chinese_text=self.is_chinese_text,
                                                                   embedding_model=self.embedding_model,
                                                                   embedding_model_name=self.embedding_model_name,
                                                                   **kwargs)
        searched_docs = web_vector_store_decorator.web_search_docs(query=self.query_str,
                                                                   search_count=self.kwargs.get("web_search_count",
                                                                                                WEB_SEARCH_COUNT))
        web_vector_store_decorator.add_documents(searched_docs)
        self.web_vector_store_decorator = web_vector_store_decorator
        self.default_retriever: ParentDocumentOfFAISSRetriever = self.web_vector_store_decorator.as_retriever(
            k=self.search_top_k,
            search_type=self.search_type)

        if self.use_embedding_filter_assist:
            from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
            embedding_similarity_threshold = kwargs.get("embedding_similarity_threshold", 0.5)
            relevant_filter = EmbeddingsFilter(embeddings=self.embedding_model,
                                               similarity_threshold=embedding_similarity_threshold)
            from langchain_community.document_transformers import EmbeddingsRedundantFilter
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_model)
            from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
            pipeline_compressor = DocumentCompressorPipeline(
                # transformers=[relevant_filter, redundant_filter]
                transformers=[relevant_filter]
            )
            from langchain_classic.retrievers import ContextualCompressionRetriever
            self.default_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                    base_retriever=self.default_retriever)
            logger.info(
                "[DefaultRetriever]-Using embedding filter assist and embedding_similarity_threshold: {}".format(
                    embedding_similarity_threshold))

        checkpoint_time1 = time.time()
        logger.info(f"[DefaultRetriever]-web_search_docs time cost: {checkpoint_time1 - start_time}")

    def get_retriever(self):
        return self.default_retriever

    def get_multi_queries(self):
        return self.multi_queries

    def housekeeping(self):
        self.web_vector_store_decorator.remove_vector_store()


class RetrieverWrapperFactory:
    """检索器包装器工厂类，负责创建对应类型的检索器包装器"""

    # 检索器类型到包装器类的映射表
    _wrapper_classes = {
        BuildKnowledgeIndexTypeEnum.LOCAL_VECTOR_STORE: LocalVectorStoreRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.HIPPO_KNOWLEDGE_GRAPH: HippoRAGRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.ELASTICSEARCH: ElasticSearchRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.WEB_SEARCH: WebSearchVectorStoreRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.MIXED_LOCAL_VECTOR_STORE_AND_WEB_SEARCH: MixedLocalVSAndWebSearchRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.MIXED_ELASTICSEARCH_AND_WEB_SEARCH: MixedElasticsearchAndWebSearchRetrieverWrapperV2,
        BuildKnowledgeIndexTypeEnum.MIXED_HIPPO_KNOWLEDGE_GRAPH_AND_WEB_SEARCH: MixedHippoKnowledgeGraphAndWebSearchRetrieverWrapperV2,
    }

    @classmethod
    def create_wrapper(cls, retriever_type, **kwargs):
        """创建指定类型的检索器包装器

        Args:
            retriever_type: 检索器类型，BuildKnowledgeIndexTypeEnum枚举值
            **kwargs: 传递给检索器包装器的参数

        Returns:
            BaseRetrieverWrapperV2: 创建的检索器包装器实例

        Raises:
            ValueError: 当指定的检索器类型不受支持时
        """
        if retriever_type not in cls._wrapper_classes:
            raise ValueError(f"Not retrieved type: {retriever_type}")

        wrapper_class = cls._wrapper_classes[retriever_type]

        # 其他类型直接创建
        return wrapper_class(**kwargs)

    @classmethod
    def register_wrapper_class(cls, retriever_type, wrapper_class):
        """注册新的检索器类型及其对应的包装器类

        Args:
            retriever_type: 检索器类型，BuildKnowledgeIndexTypeEnum枚举值
            wrapper_class: 对应的包装器类
        """
        cls._wrapper_classes[retriever_type] = wrapper_class


class DefaultRetrieverV2:
    def __init__(self,
                 **kwargs) -> None:
        '''
        :param query_str: the query string to be used for retrieval
        :param llm: the LLM to be used for query generation
        :param is_chinese_text: whether the text is Chinese
        :param retriever_type: the type of retriever to be used. Must be one of "local_vector_stores", "web_search", or "fixed".
                And fixed means a combination of local_vector_stores and web_search
        :param local_vs_folder_name: the folder name of the local vector store to be used
        :param search_type: the type of search to be used. Must be one of "mmr" or "similarity"
        :param search_top_k: the number of documents to be retrieved
        :param use_embedding_filter_assist: whether to use embedding filter assist
        :param use_reorder_assist: whether to use reorder assist
        :param kwargs: other keyword arguments: fixed_weights, search_tool, ignore_content_len, is_jupyter_notebook, embedding_similarity_threshold.
                And fixed_weights is a list of two floats that sum to 1, means the weights of web_search: local_vector_stores .
        '''
        query_str = kwargs.get("query_str", None)
        # assert query_str is not None, "query_str must be provided"
        self.query_str = query_str
        self.search_top_k = kwargs.get("search_top_k", 4)
        self.use_reorder_assist = kwargs.get("use_reorder_assist", False)

        self.default_retriever: BaseRetriever

        retriever_type = kwargs.get("retriever_type", BuildKnowledgeIndexTypeEnum.WEB_SEARCH)
        kwargs_copy = kwargs.copy()
        if "retriever_type" in kwargs_copy:
            kwargs_copy.pop("retriever_type")
        self.default_retriever_wrapper = RetrieverWrapperFactory.create_wrapper(
            retriever_type=retriever_type,
            **kwargs_copy
        )

        self.use_analysis_enhancement = kwargs.get("use_analysis_enhancement", True)
        if self.use_analysis_enhancement:
            knowledge_index_folder_name = kwargs.get("knowledge_index_folder_name") or kwargs.get(
                "local_vs_folder_name")
            assert knowledge_index_folder_name is not None, "knowledge_index_folder_name must be provided when use_analysis_enhancement is True"

            # 用装饰器包装原始检索器
            from core.data_connections.retrievers.document_analysis_retriever.analysis_document_retriever import \
                AnalysisEnhancedRetrieverDecorator
            self.default_retriever = AnalysisEnhancedRetrieverDecorator(
                base_retriever_wrapper=self.default_retriever_wrapper,
                knowledge_index_folder_name=knowledge_index_folder_name,
            )
        else:
            self.default_retriever = self.default_retriever_wrapper.get_retriever()

    def get_relevant_documents(self, query, top_k: int = None) -> List[Document]:
        if self.query_str is None:
            self.query_str = query
        if top_k is not None:
            self.search_top_k = top_k

        relevant_docs = self.default_retriever.invoke(input=self.query_str, k=self.search_top_k)
        if self.use_reorder_assist:
            from langchain_community.document_transformers import LongContextReorder
            reordering = LongContextReorder()
            # logger.info("Previous relevant_docs: {}".format(relevant_docs))
            relevant_docs = reordering.transform_documents(relevant_docs)
            # logger.info("Reordered relevant_docs: {}".format(relevant_docs))
            logger.debug(f"use_reorder_assist for query:{query}")

        if len(relevant_docs) >= self.search_top_k:
            relevant_docs = relevant_docs[:self.search_top_k]

        threading.Thread(target=self.housekeeping, name="housekeep_nwvs_thread").start()
        return relevant_docs

    def get_multi_queries(self):
        return self.default_retriever_wrapper.get_multi_queries()

    def get_search_keywords(self):
        return self.query_str

    def housekeeping(self):
        if self.default_retriever_wrapper is not None:
            self.default_retriever_wrapper.housekeeping()