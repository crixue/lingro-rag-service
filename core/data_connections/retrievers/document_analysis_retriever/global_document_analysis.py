import os
from os import PathLike

from langchain_core.documents import Document

from configs.model_config import KB_ROOT_PATH, logger, CHARACTER_TEXT_SPLITTER, \
    MINIMAL_DOCUMENT_ANALYSIS_CHUNK_SIZE_RATIO
from controllers.common.common_constant import DOC_STORE_SUFFIX
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.data_connections.retrievers.storage.documents_storage import GlobalDocumentContextStorage
from json_schema_definitions import JsonSchemaDefinition
from prompt_definitions import PromptDefinition
from tools_definitions.tools.summarize_tools import summary_file_content
from utils.lang_util import lang_def_name


class GlobalDocumentAnalysis:
    def __init__(self,
                 analysis_result_folder_name: str = None,
                 is_chinese_text: bool = True,
                 ):
        assert analysis_result_folder_name is not None, "folder_name should not be None"
        self.folder_name = analysis_result_folder_name
        self.doc_store_path = os.path.join(KB_ROOT_PATH, self.folder_name + DOC_STORE_SUFFIX)
        if not os.path.exists(self.doc_store_path):
            os.makedirs(self.doc_store_path)
        self.document_storage = GlobalDocumentContextStorage(root_path=self.doc_store_path)
        self.system_prompt = PromptDefinition.load_prompt_json_dict_static(
            lang=lang_def_name(is_chinese_text=is_chinese_text),
            prompt_name="analysis_document_system_prompt").get("template", "")
        self.response_json_schema = JsonSchemaDefinition.load_builtin_json_schema_dict_static(
            lang=lang_def_name(is_chinese_text=is_chinese_text),
            schema_name="analysis_document_json_schema")
        self.chunk_size = CHARACTER_TEXT_SPLITTER["cn_chunk_size"] if is_chinese_text else CHARACTER_TEXT_SPLITTER["en_chunk_size"]
        self.is_chinese_text = is_chinese_text

    async def analysis_documents_and_save_async(self,
                                                user_id: str,
                                                document_path_list: list[tuple[str, str|PathLike]],
                                                *,
                                                require_analysis: bool = False,
                                                **kwargs
                                                ):
        """
        全局文档解析方法
        :param document_path_list: list[tuple[str, str|PathLike]] 是 (document_id,文档路径) Tuple 列表
        :return: dict, 解析结果
        """
        assert user_id is not None, "user_id should not be None"
        assert document_path_list is not None and len(
            document_path_list) > 0, "document_path_list should not be None or empty"

        result_docs = []
        for document_id, document_path in document_path_list:
            if not os.path.exists(document_path):
                logger.warning(f"[analysis_documents_and_save_async]-document_path not exists: {document_path}")
                continue

            file_loader = DefaultFileLoaderComponent(file_path=document_path)
            doc = file_loader.load_full_content(include_metadata=True)
            if doc is None or not doc.page_content:
                logger.warning(f"[analysis_documents_and_save_async]-load_full_content failed: {document_path}, content is empty or None")
                continue

            # Check if the summary document is exists and valid?
            check_docs = self.document_storage.retriever_documents([document_id])
            if check_docs and len(check_docs) > 0 and not require_analysis:
                logger.info(f"[analysis_documents_and_save_async]-Document already analyzed, skip: {document_path}")
                continue

            try:
                json_result = summary_file_content(
                    file_path=document_path,
                    user_id=user_id,
                    is_chinese_text=self.is_chinese_text,
                    reduce_phase_template_without_metadata=False
                )

                if json_result:
                    result_doc = Document(
                        id=document_id,
                        page_content=json_result
                    )
                    result_docs.append(result_doc)
                    logger.info(f"[analysis_documents_and_save_async]-Successfully analyzed document: {document_path}")
                else:
                    logger.warning(f"[analysis_documents_and_save_async]-No result from summary_file_content: {document_path}")
            except Exception as e:
                logger.exception(f"[analysis_documents_and_save_async]-Failed to analyze document {document_path}: {e}")

        # 保存解析结果到文档存储
        if result_docs:
            self.document_storage.save_documents(result_docs)
            logger.info(f"[analysis_documents_and_save_async]-Saved {len(result_docs)} analyzed documents")
