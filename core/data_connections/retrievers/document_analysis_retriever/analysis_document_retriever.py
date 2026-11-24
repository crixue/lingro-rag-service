import json
import os
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from configs.model_config import logger, KB_ROOT_PATH
from controllers.common.common_constant import MetadataConstants, DOC_STORE_SUFFIX
from core.common.common_constant import DocumentAnalysisConstants
from core.data_connections.retrievers.base_retriever_wrapper_v2 import BaseRetrieverWrapperV2
from core.data_connections.retrievers.storage.documents_storage import GlobalDocumentContextStorage


class AnalysisEnhancedRetrieverDecorator(BaseRetriever):
    base_retriever: Optional[BaseRetriever] = None
    enhancement_mode: str = Field(default="eager")  # eager/lazy
    global_doc_analysis: Optional[GlobalDocumentContextStorage] = None

    def __init__(self, base_retriever_wrapper: BaseRetrieverWrapperV2, knowledge_index_folder_name: str = None,
                 enhancement_mode: str = "eager", *args: Any, **kwargs: Any):  # lazy/eager
        super().__init__(*args, **kwargs)
        self.base_retriever = base_retriever_wrapper.get_retriever()
        self.enhancement_mode = enhancement_mode
        doc_store_path = os.path.join(KB_ROOT_PATH, knowledge_index_folder_name + DOC_STORE_SUFFIX)
        self.global_doc_analysis = GlobalDocumentContextStorage(root_path=doc_store_path)

    def _pre_process(self):
        pass

    def _post_enhancement(self, docs: list[Document]) -> list[Document] | None:
        if docs is None or len(docs) <= 0:
            return docs

        # 经过split之后的文档，doc id有两个部分组成，中间使用_分割，前半部分是原始文档id，后半部分是split之后的编号
        original_doc_ids = set()
        for doc in docs:
            original_doc_id = doc.metadata.get("doc_id", None)
            if original_doc_id is not None and "_" in original_doc_id:
                original_doc_id = original_doc_id.split("_")[0]
            if original_doc_id is not None:
                original_doc_ids.add(original_doc_id)

        analysis_docs = self.global_doc_analysis.retriever_documents(ids=list(original_doc_ids))

        # 如果没有分析文档，直接返回原始文档
        if not analysis_docs or len(analysis_docs) == 0:
            return docs

        # 构建分析文档字典，以原始文档ID为键
        analysis_dict = {}
        for analysis_doc in analysis_docs:
            try:
                # 解析分析文档的JSON内容
                analysis_content = json.loads(analysis_doc.page_content)
                analysis_dict[analysis_doc.id] = analysis_content
            except Exception as e:
                logger.warning(f"[_get_relevant_documents] Failed to parse analysis document {analysis_doc.id}: {e}")
                continue

        # 将分析信息注入到对应文档的metadata中
        for doc in docs:
            original_doc_id = doc.metadata.get("doc_id", None)
            if original_doc_id is not None and "_" in original_doc_id:
                original_doc_id = original_doc_id.split("_")[0]

            if original_doc_id and original_doc_id in analysis_dict:
                analysis_info = analysis_dict[original_doc_id]

                # 注入分析信息到metadata
                if DocumentAnalysisConstants.CORE_SUMMARY in analysis_info:
                    doc.metadata[MetadataConstants.ANALYSIS_GLOBAL_CORE_SUMMARY] = analysis_info[
                        DocumentAnalysisConstants.CORE_SUMMARY
                    ]

                if DocumentAnalysisConstants.THEME_TAGS in analysis_info:
                    doc.metadata[MetadataConstants.ANALYSIS_GLOBAL_THEME_TAGS] = analysis_info[
                        DocumentAnalysisConstants.THEME_TAGS
                    ]

                if DocumentAnalysisConstants.DOCUMENT_INTENT_TYPES in analysis_info:
                    doc.metadata[MetadataConstants.ANALYSIS_GLOBAL_DOCUMENT_INTENT_TYPES] = analysis_info[
                        DocumentAnalysisConstants.DOCUMENT_INTENT_TYPES
                    ]

                if DocumentAnalysisConstants.KEY_ENTITIES in analysis_info:
                    doc.metadata[MetadataConstants.ANALYSIS_GLOBAL_KEY_ENTITIES] = analysis_info[
                        DocumentAnalysisConstants.KEY_ENTITIES
                    ]
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> list[Document] | None:
        docs = self.base_retriever.invoke(query, run_manager=run_manager, **kwargs)

        if self.enhancement_mode == "eager":
            try:
                docs = self._post_enhancement(docs)
            except Exception as e:
                logger.exception(f"[_get_relevant_documents] Enhancement failed: {e}")
        else:
            pass  # lazy mode, do nothing now

        return docs
