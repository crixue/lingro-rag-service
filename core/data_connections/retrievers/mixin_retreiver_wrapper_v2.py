import asyncio
import os
from abc import abstractmethod, ABC
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from configs.model_config import logger, NEO4J_CONFIG, KB_ROOT_PATH, WEB_SEARCH_COUNT
from core.data_connections.retrievers.base_retriever_wrapper_v2 import BaseRetrieverWrapperV2
from core.data_connections.retrievers.elasticsearch_retriever.elasticsearch_retriever import \
    document_field_mapper, DefaultElasticsearchRetriever
from core.data_connections.retrievers.knowlege_graph_retrievers.hippo_retriever import HippoRetriever
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import GRAPH_DB_PREFIX
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.hipporag_with_graph_db import HippoRAG
from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import CustomLocalVectorDecorator
from core.data_connections.retrievers.network_retrievers.web_search_vector_store import WebSearchVectorStoreDecorator
from core.data_connections.retrievers.storage.documents_storage import DefaultDocumentsStorage
from core.data_connections.retrievers.utils.retrieve_util import parse_query, PARSE_INFO_KEY_PHRASES, \
    PARSE_INFO_KEY_LIMITATION, PARSE_INFO_KEY_SYNTAX
from utils.elastic_search_client import ElasticSearchClient
from utils.neo4j_client import Neo4jGraphDBClient


class BaseMixedRetrieverWrapperV2(BaseRetrieverWrapperV2, ABC):
    """
    混合检索器的抽象基类，用于处理多种检索器类型的组合
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.fixed_weights = kwargs.get("fixed_weights", [0.5, 0.5])
        if len(self.fixed_weights) != 2 or self.fixed_weights[0] + self.fixed_weights[1] != 1:
            raise ValueError("fixed_weights必须是两个浮点数的列表，且和为1")

        # 创建主检索器和辅助检索器的异步任务
        primary_retriever_task = self.loop.create_task(self._create_primary_retriever_async())
        secondary_retriever_task = self.loop.create_task(self._create_secondary_retriever_async())

        # 组合检索器
        self.loop.run_until_complete(self._ensemble_retriever_async(
            primary_retriever_task, secondary_retriever_task, self.fixed_weights))

        # 应用嵌入过滤辅助（如果启用）
        if self.use_embedding_filter_assist:
            self._apply_embedding_filter()

    async def _create_primary_retriever_async(self) -> BaseRetriever:
        """创建网络搜索检索器"""
        vector_store_decorator = WebSearchVectorStoreDecorator(
            is_chinese_text=self.is_chinese_text,
            embedding_model=self.embedding_model,
            embedding_model_name=self.embedding_model_name,
            **self.kwargs
        )
        searched_docs = vector_store_decorator.web_search_docs(
            query=self._parse_query_str_for_websearch(self.query_str),
            search_count=self.kwargs.get("web_search_count", WEB_SEARCH_COUNT // 2)
        )
        vector_store_decorator.add_documents(searched_docs)
        return vector_store_decorator.as_retriever(
            k=self.search_top_k,
            search_type=self.search_type
        )

    @abstractmethod
    async def _create_secondary_retriever_async(self) -> BaseRetriever:
        """创建辅助检索器（抽象方法）"""
        pass

    async def _ensemble_retriever_async(self,
                                        primary_retriever_task: asyncio.Task[BaseRetriever],
                                        secondary_retriever_task: asyncio.Task[BaseRetriever],
                                        weights: List[float]) -> None:
        """组合两个检索器"""
        primary_retriever, secondary_retriever = await asyncio.gather(
            primary_retriever_task, secondary_retriever_task)
        self.primary_retriever = primary_retriever
        self.secondary_retriever = secondary_retriever
        self.default_retriever = EnsembleRetriever(
            retrievers=[primary_retriever, secondary_retriever],
            weights=weights)

    def _parse_query_str_for_websearch(self, query_str: str) -> str:
        """解析查询字符串以适应网络搜索"""
        return query_str.strip() if query_str else ""

    def _apply_embedding_filter(self):
        """应用嵌入过滤辅助"""
        from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
        from langchain_community.document_transformers import EmbeddingsRedundantFilter
        from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
        from langchain_classic.retrievers import ContextualCompressionRetriever

        embedding_similarity_threshold = self.kwargs.get("embedding_similarity_threshold", 0.5)
        relevant_filter = EmbeddingsFilter(
            embeddings=self.embedding_model,
            similarity_threshold=embedding_similarity_threshold
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_model)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[relevant_filter, redundant_filter]
        )
        self.default_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=self.default_retriever
        )
        logger.debug(f"使用嵌入过滤辅助，相似度阈值: {embedding_similarity_threshold}")

    def get_retriever(self):
        return self.default_retriever

    def get_multi_queries(self):
        return self.multi_queries

    def housekeeping(self):
        """清理资源"""
        if hasattr(self, 'primary_retriever') and hasattr(self.primary_retriever, 'remove_vector_store'):
            self.primary_retriever.remove_vector_store()
        if hasattr(self, 'secondary_retriever') and hasattr(self.secondary_retriever, 'remove_vector_store'):
            self.secondary_retriever.remove_vector_store()
        logger.info("[BaseMixedRetrieverWrapper] cleaned over")


class MixedLocalVSAndWebSearchRetrieverWrapperV2(BaseMixedRetrieverWrapperV2, ABC):
    """本地向量存储与网络搜索的混合检索器"""
    async def _create_secondary_retriever_async(self) -> BaseRetriever:
        """创建本地向量存储检索器"""
        vector_store_decorator = CustomLocalVectorDecorator(
            folder_name=self.knowledge_index_folder_name,
            is_chinese_text=self.is_chinese_text,
            embedding_model=self.embedding_model,
            embedding_model_name=self.embedding_model_name,
            **self.kwargs
        )
        if not vector_store_decorator.exists_local_faiss_indies():
            error_msg = f"local_vs_path : {vector_store_decorator.current_vs_path} faiss index not exists!"
            logger.error(f"[DefaultRetriever]-[mixed type] {error_msg}")
            raise ValueError(error_msg)

        return vector_store_decorator.as_retriever(
            is_chinese_text=self.is_chinese_text,
            search_type=self.search_type,
            k=self.search_top_k
        )


class MixedElasticsearchAndWebSearchRetrieverWrapperV2(BaseMixedRetrieverWrapperV2, ABC):
    """Elasticsearch与网络搜索的混合检索器"""
    async def _create_secondary_retriever_async(self) -> BaseRetriever:
        """创建Elasticsearch检索器"""
        assert self.knowledge_index_folder_id is not None, "knowledge_index_folder_id must be provided"
        assert self.version is not None, "version must be provided"

        top_k = self.search_top_k if self.search_top_k is not None else self.kwargs.get("top_k", 5)
        return DefaultElasticsearchRetriever(
            es_client=ElasticSearchClient(embedding_model=self.embedding_model).es_client,
            index_name=self.knowledge_index_folder_id + "ver" + str(self.version),
            embedding_model=self.embedding_model,
            document_mapper=document_field_mapper,
            query_kwargs={
                "top_k": top_k,
                "num_candidates": self.kwargs.get("num_candidates", 50)
            }
        )

    def _parse_query_str_for_websearch(self, query_str: str) -> str:
        """
        暂时使用 parse_query 方法来解析 query 字符串，未来web search 支持，可以将limitations、syntax_items等复杂的解析逻辑转换成websearch的搜索逻辑。
        """
        parsed_query_info = parse_query(query_str)
        phrases = parsed_query_info.get(PARSE_INFO_KEY_PHRASES, [])
        limitations = parsed_query_info.get(PARSE_INFO_KEY_LIMITATION, [])
        syntax_items = parsed_query_info.get(PARSE_INFO_KEY_SYNTAX, [])

        parsed_query = query_str  # Default to the original query
        if phrases and len(phrases) > 0:
            parsed_query = " ".join(phrases)
        return parsed_query


class MixedHippoKnowledgeGraphAndWebSearchRetrieverWrapperV2(BaseMixedRetrieverWrapperV2, ABC):
    """知识图谱与网络搜索的混合检索器"""
    async def _create_secondary_retriever_async(self) -> BaseRetriever:
        """创建知识图谱检索器"""
        assert self.knowledge_index_folder_id is not None, "knowledge_index_folder_id must be provided"
        assert self.version is not None, "version must be provided"

        top_k = self.search_top_k if self.search_top_k is not None else self.kwargs.get("top_k", 4)
        dataset = self.kwargs.get("file_name", "default")
        neo4j_client = self.kwargs.get("neo4j_client", None) or Neo4jGraphDBClient(
            uri=self.kwargs.get("neo4j_uri", NEO4J_CONFIG["NEO4J_URI"]),
            user=self.kwargs.get("neo4j_user", NEO4J_CONFIG["NEO4J_USER"]),
            password=self.kwargs.get("neo4j_password", NEO4J_CONFIG["NEO4J_PASSWORD"]),
        )
        graph_db_name = GRAPH_DB_PREFIX + self.knowledge_index_folder_id + "ver" + str(self.version)
        save_path = os.path.join(KB_ROOT_PATH, f"{self.knowledge_index_folder_name}_doc_store")
        storage = DefaultDocumentsStorage(root_path=save_path)
        self.storage = storage

        hippo_rag = HippoRAG(
            corpus_name=dataset,
            llm_client=self.llm,
            retriever_name=self.embedding_model_name,
            force_retrieve=self.kwargs.get("force_retrieve", False),
            local_doc_folder_name=self.knowledge_index_folder_name,
            checkout_version=self.version,
            graph_db_name=graph_db_name,
            neo4j_client=neo4j_client,
            embedding_model=self.embedding_model,
            doc_ensemble=True,
            dpr_only=False,
            user_folder_id=self.knowledge_index_folder_id,
            storage=storage,
        )

        return HippoRetriever(
            hippo_rag=hippo_rag,
            docstore=storage,
            search_kwargs={"top_k": top_k},
            score_threshold=self.kwargs.get("score_threshold", 0.09),
            enable_logging=self.kwargs.get("enable_logging", False)
        )

    def _parse_query_str_for_websearch(self, query_str: str) -> str:
        """
        Parse the query string for web search.
        Args:
            query: The query string
        Returns:
            Parsed query string for web search.
        """
        parsed_query_info = parse_query(query_str)
        phrases = parsed_query_info.get("phrases", [])
        parsed_query = query_str  # Default to the original query
        if phrases and len(phrases) > 0:
            parsed_query = " ".join(phrases)
        return parsed_query