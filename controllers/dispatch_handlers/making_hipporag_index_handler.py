import os
import shutil
from typing import List, Optional

from langchain_core.documents import Document

from configs.model_config import logger, EXTRACT_OPENIE_FROM_DOCS_PROCESSOR_NUM, NEO4J_CONFIG, \
    CACHE_ROOT_PATH, KB_ROOT_PATH
from controllers.common.retry_making_index_context import CreateIndexRetryableContext, retryable_step
from controllers.dispatch_handlers.base_making_index_handler import BaseBuildingIndexHandler
from controllers.enums.create_index_steps_enum import CreateHippoRAGStepsEnum
from core.common.common_util import documents_id_filling
from core.data_connections.retrievers.knowlege_graph_retrievers.hippo_retriever import HippoRetriever
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import GRAPH_DB_PREFIX
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.hipporag_with_graph_db import HippoRAG
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.openie_with_retrieval_option_parallel import \
    OpenIEWithRetrievalOption, remove_nodes_rels_from_graphdb_by_doc_ids
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.retrieval_module import RetrievalModule
from core.data_connections.retrievers.storage.documents_storage import DefaultDocumentsStorage
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from utils.neo4j_client import Neo4jGraphDBClient


class DefaultBuildingHippoRAGGraphHandler(BaseBuildingIndexHandler):
    def __init__(self,
                 neo4j_graph_db_name: str = None,
                 is_chinese_text: bool = False,
                 **kwargs):
        '''
            Making vector store for file types.
            :param kwargs: Pls refer to BaseUnstructuredFileLoader
        '''
        super().__init__(is_chinese_text=is_chinese_text, **kwargs)
        assert neo4j_graph_db_name is not None, "neo4j_graph_db_name should not be None"
        assert self.id is not None, "id should not be None"
        self.storage = kwargs.get("storage", None)
        if self.storage is None:
            assert self.local_doc_folder_name is not None, "local_doc_folder_name should not be None"
            save_path = os.path.join(KB_ROOT_PATH, f"{self.local_doc_folder_name}_doc_store")
            self.storage = DefaultDocumentsStorage(document_id_prefix=self.document_id_prefix, root_path=save_path)
            logger.info(f"[DefaultBuildingHippoRAGGraphHandler]-Storage path:{save_path}")
        if self.user_folder_id:
            self.retryable_context = kwargs.get("retryable_context", CreateIndexRetryableContext(
                steps_enum=CreateHippoRAGStepsEnum,
                current_step= CreateHippoRAGStepsEnum.INITIATE.name,
                id=self.id,
                user_folder_id=self.user_folder_id))
        self.graph_db_name = neo4j_graph_db_name
        self.neo4j_client = kwargs.get("neo4j_client", None) or Neo4jGraphDBClient(
            uri=kwargs.get("neo4j_uri", NEO4J_CONFIG["NEO4J_URI"]),
            user=kwargs.get("neo4j_user", NEO4J_CONFIG["NEO4J_USER"]),
            password=kwargs.get("neo4j_password", NEO4J_CONFIG["NEO4J_PASSWORD"]),
        )

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.INITIATE.name)
    def _create_neo4j_graph_db(self, openIE_client: OpenIEWithRetrievalOption, **kwargs):
        openIE_client.create_neo4j_graph_db(neo4j_client=self.neo4j_client, graph_db_name=self.graph_db_name)

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.LOAD_DOCS_AND_SAVE_TO_STORAGE.name)
    def _load_docs_and_save_to_storage(self, docs: Optional[List[Document]] = None, **kwargs) -> list[Document]:
        if docs is None:
            docs: List[Document] = self.load_docs(**kwargs)
            docs = documents_id_filling(documents=docs, document_id_prefix=self.document_id_prefix)

        docs = self.inject_metadata(docs, metadata=kwargs.get("metadata", {}))
        docs = self.storage.save_documents(docs, require_document_id_filling=False)
        # docs = replace_words_on_doc_page_content(docs)
        self.docs_index_count = len(docs)
        logger.info(f"[_load_docs_and_save_to_storage]-Docs count: {self.docs_index_count}")
        self.retryable_context_info.update({"docs": docs})
        return docs

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.EXTRACT_OPENIE_FROM_DOCUMENT.name)
    def _extract_openie_from_document(self, openIE_client: OpenIEWithRetrievalOption, docs: List[Document],
                                      **kwargs) -> list[tuple]:
        logger.info(f"[_extract_openie_from_document]-Extracting openie from docs...")
        extracted_openie_list, lm_total_tokens = openIE_client.extract_openie_from_document(docs,
                                                                                            num_processes=EXTRACT_OPENIE_FROM_DOCS_PROCESSOR_NUM,
                                                                                            is_chinese_text=self.is_chinese_text,
                                                                                            # num_passages=1,
                                                                                            num_passages='all',
                                                                                            id=self.id
                                                                                            )
        self.retryable_context_info.update({"extracted_openie_list": extracted_openie_list})
        self.retryable_context_info.pop("docs", None)
        return extracted_openie_list

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.SAVE_EXTRACT_OPENIE_2_NEO4J_GRAPH.name)
    def _save_extract_openie_2_neo4j_graph(self, extracted_openie_list: list[tuple],
                                           openIE_client: OpenIEWithRetrievalOption, **kwargs):
        logger.info(f"[_save_extract_openie_2_neo4j_graph]-Saving openie to graphdb-{self.graph_db_name}...")
        openIE_client.save_extract_openie_2_neo4j_graph(extracted_openie_list,
                                                        neo4j_client=self.neo4j_client,
                                                        graph_db_name=self.graph_db_name)

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.AUGMENT_GRAPHDB_WITH_SIMILARITY.name)
    def _augment_graphdb_with_similarity(self, **kwargs):
        retrieve_module = RetrievalModule(self.embedding_model_name,
                                          embedding_model=self.embedding_model,
                                          neo4j_client=self.neo4j_client,
                                          graph_db_name=self.graph_db_name, )
        similarity_threshold = kwargs.get("similarity_threshold", 0.85)
        similarity_pick_nums = kwargs.get("similarity_pick_nums", 20)
        logger.info(f"[_augment_graphdb_with_similarity]-Augmenting graphdb with similarity threshold: "
                    f"{similarity_threshold} and pick nums: {similarity_pick_nums}...")
        retrieve_module.augment_graphdb_with_similarity(threshold=similarity_threshold,
                                                        similarity_pick_nums=similarity_pick_nums)

    def _init_rag(self, dataset: str, **kwargs):
        rag = HippoRAG(corpus_name=dataset,
                       llm_client=kwargs.get("llm_client", None),
                       retriever_name=self.embedding_model_name,
                       force_retrieve=kwargs.get("force_retrieve", True),
                       local_doc_folder_name=self.local_doc_folder_name,
                       checkout_version=self.checkout_version,
                       graph_db_name=self.graph_db_name,
                       neo4j_client=self.neo4j_client,
                       embedding_model=self.embedding_model,
                       doc_ensemble=True, dpr_only=False,
                       is_chinese_text=self.is_chinese_text,
                       user_folder_id=self.user_folder_id,
                       storage=self.storage,
                       )
        return rag

    @retryable_step(steps_enum=CreateHippoRAGStepsEnum, step=CreateHippoRAGStepsEnum.PRE_BUILT_RETRIEVE_INDEX.name)
    def _pre_built_retrieve_index(self, *, dataset: str, **kwargs):
        rag = self._init_rag(dataset=dataset, **kwargs)
        logger.info(f"[create_index_and_save_to_storage]-Pre-built retrieve index...")
        rag.pre_built_retrieve_index()
        doc_retriever = HippoRetriever(hippo_rag=rag, docstore=self.storage,
                                       search_kwargs={"top_k": kwargs.get("top_k", 4)},
                                       k=kwargs.get("score_threshold", 0.1))
        return doc_retriever

    def precompute_and_save_results(self, docs: Optional[List[Document]] = None,
                                    user_id: str = None,
                                    product_id: str = "MODEL-0002",
                                    **kwargs) -> HippoRetriever:
        '''
            Use ParentDocumentRetriever add documents to local vector store

            :param kwargs:
                file_name: str
        '''
        assert self.checkout_version is not None, "checkout_version should not be None"

        assert user_id is not None, "user_id should not be None"
        assert product_id is not None, "product_id should not be None"
        model_dict = SupportedModelsInfoUtil.find_supported_model_by_id(product_id)
        product_id = model_dict["id"]
        model_name = model_dict["name"]
        llm_model_config = kwargs.get("llm_model_config", {
            "temperature": 0,
            "top_p": 0.9,
            "streaming": False})
        llm_client = DefaultLangchainLLM.llm_client_factory(llm_model_name=model_name,
                                                            llm_model_config=llm_model_config,
                                                            user_id=user_id,
                                                            product_id=product_id, )
        dataset = kwargs.get("file_name", "default")
        openIE_client = OpenIEWithRetrievalOption(
            dataset=dataset, chat_llm_client=llm_client,
            retriever_name=self.embedding_model_name, is_chinese_text=self.is_chinese_text,
            model_name=model_name,
        )
        self.retryable_request_params = {
            "docs": docs,
            "user_id": user_id,
            "product_id": product_id,
            **kwargs,
        }

        self._create_neo4j_graph_db(openIE_client=openIE_client, **kwargs)

        docs = self._load_docs_and_save_to_storage(docs, **kwargs)

        extracted_openie_list = self._extract_openie_from_document(openIE_client=openIE_client, docs=docs,
                                                                   llm_client=llm_client, model_name=model_name,
                                                                   user_id=user_id, product_id=product_id,
                                                                   **kwargs)

        self._save_extract_openie_2_neo4j_graph(extracted_openie_list=extracted_openie_list,
                                                openIE_client=openIE_client,
                                                **kwargs)

        self._augment_graphdb_with_similarity(**kwargs)

        retriever = self._pre_built_retrieve_index(dataset=dataset, llm_client=llm_client, **kwargs)
        self._housekeeping_precompute_index_cache()
        return retriever

    def del_part_of_indies(self, ids: List[str], **kwargs):
        """
        1. Remove nodes and relationships from graphdb
        2. Remove docs from docstore
        3. Delete retryable context info
        4. Delete user cache folder
        """
        # remove from graphdb
        remove_nodes_rels_from_graphdb_by_doc_ids(ids, self.neo4j_client, self.graph_db_name)
        # remove docs from docstore
        self.storage.delete_documents(ids)
        self.retryable_context.delete_context_info()

        dataset = kwargs.get("file_name", "default")
        self._pre_built_retrieve_index(dataset=dataset, force_retrieve=True, **kwargs)

    def del_version_of_indies_and_docs(self, **kwargs):
        """
        1. 删除图数据库对应的schema
        2. 删除docstore storage 中的所有文档
        3. 递归删除kb user folder下所有内容
        """
        with self.neo4j_client.driver.session() as session:
            session.run(f"DROP DATABASE {self.graph_db_name} IF EXISTS")
            logger.info(f"[del_all_indies_and_docs]-Drop graphdb-{self.graph_db_name} successfully.")
        self.storage.delete_storage_documents()
        output_dir = os.path.join(kwargs.get('kb_output_dir', CACHE_ROOT_PATH), self.local_doc_folder_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"[del_all_indies_and_docs]-Delete user cache folder {output_dir} successfully.")

    def del_workspace(self, **kwargs):
        graph_db_prefix = kwargs.get("graph_db_prefix", GRAPH_DB_PREFIX)
        graph_names = self.neo4j_client.find_graphs_with_prefix(graph_db_prefix + self.user_folder_id)
        if graph_names:
            for graph_name in graph_names:
                with self.neo4j_client.driver.session() as session:
                    session.run(f"DROP DATABASE {graph_name} IF EXISTS")
                    logger.info(f"[del_workspace]-Drop graphdb-{graph_name} successfully.")
        output_dir = os.path.join(kwargs.get('kb_output_dir', CACHE_ROOT_PATH), self.user_folder_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"[del_all_indies_and_docs]-Delete user cache folder {output_dir} successfully.")
