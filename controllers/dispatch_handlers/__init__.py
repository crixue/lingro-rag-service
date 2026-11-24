import asyncio
import glob
import json
import os
import pickle
import shutil
from abc import ABC, abstractmethod

from api_routers.context import set_user_id
from configs.model_config import logger, MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR, \
    CREATE_INDEX_RETRYABLE_CONTEXT_FILE_DIR_NAME, KB_ROOT_PATH
from controllers.common.common_constant import MetadataConstants
from controllers.common.retry_making_index_context import CreateIndexRetryableContext
from controllers.dispatch_handlers.base_making_index_handler import BaseBuildingIndexHandler
from controllers.dispatch_handlers.making_elastic_search_index_handler import DefaultBuildingElasticSearchIndexHandler
from controllers.dispatch_handlers.making_hipporag_index_handler import DefaultBuildingHippoRAGGraphHandler
from controllers.dispatch_handlers.making_vector_store_index_handler import DefaultBuildingVectorStoreHandler
from controllers.enums.create_index_steps_enum import CreateHippoRAGStepsEnum, CreateElasticSearchIndexStepsEnum, \
    CreateVectorStoreIndexStepsEnum
from controllers.enums.make_knowledge_base_enum import BuildKnowledgeIndexStatusEnum
from controllers.enums.source_type_enum import SourceTypeEnum
from controllers.vo.build_knowledge_index_request_vo import BuildKnowledgeIndexItemRequestVO
from controllers.vo.build_knowledge_index_request_vo import DeleteKnowledgeBaseIndexRequestVO
from core.common.common_util import generate_document_ids, generate_sub_document_ids
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.data_connections.retrievers.document_analysis_retriever.global_document_analysis import GlobalDocumentAnalysis
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import GRAPH_DB_PREFIX
from exceptions.custom_exception import RetryableStepFailedException
from core.model_io.support_models_info_util import SupportedModelsInfoUtil
from utils.common_util import detect_encoding_and_convert_file, get_beijing_current_time, get_file_name_from_path, \
    detect_if_is_chinese_text
from utils.mysql_client import CustomMysqlClient


class BaseBuildingIndexStrategy(ABC):
    def __init__(self, **kwargs):
        self.mysql_client = CustomMysqlClient()
        self.cursor = self.mysql_client.get_cursor()
        self.handler: BaseBuildingIndexHandler = None

    def _preprocess_request(self, request: BuildKnowledgeIndexItemRequestVO,
                            **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        """
        Preprocesses the request before execution.
        Override this method in subclasses to add specific preprocessing logic.
        """
        # set user id in context
        set_user_id(request.user_id)
        logger.info(f"Preprocessing request: {request.id} by user: {request.user_id}")

        # Initialize config if not present
        if not request.build_knowledge_index_config:
            request.build_knowledge_index_config = {}

        # Detect JSON/JSONL files and add appropriate configuration
        if request.save_to_path:
            file_path = request.save_to_path.lower()
            if file_path.endswith(".json"):
                request.build_knowledge_index_config["jq_schema"] = "."
                request.build_knowledge_index_config["text_content"] = False
                logger.info(f"Detected JSON file, added JSON-specific config for {request.id}")
            elif file_path.endswith(".jsonl"):
                request.build_knowledge_index_config["jq_schema"] = "."
                request.build_knowledge_index_config["json_lines"] = True
                request.build_knowledge_index_config["text_content"] = False
                logger.info(f"Detected JSONL file, added JSONL-specific config for {request.id}")

            # 需要明确当前文件是否为中文文本，方便后续选择合适的分词大小
            file_loader = DefaultFileLoaderComponent(file_path=request.save_to_path)
            if file_loader:
                detect_document = file_loader.load_size_of_content()
                if detect_document and detect_document.page_content:
                    is_chinese_text = detect_if_is_chinese_text(detect_document.page_content)
                    request.is_chinese_text = is_chinese_text

        return request

    async def _analyze_document_async(self, request: BuildKnowledgeIndexItemRequestVO):
        """
        Analyzes the document if enabled.
        """
        analysis_tool = GlobalDocumentAnalysis(
            analysis_result_folder_name=request.knowledge_index_folder_name,
            is_chinese_text=request.is_chinese_text
        )
        document_id = f"{request.id}"
        await analysis_tool.analysis_documents_and_save_async(
            user_id=request.user_id,
            document_path_list=[(document_id, request.save_to_path)],
            require_analysis=True
        )
        logger.info(
            f"[_analyze_document]-Document analysis completed for request: {request.id}, results stored in: {request.knowledge_index_folder_name}")

    async def execute_with_preprocessing_async(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        """
        Executes the strategy with preprocessing.
        """
        # Preprocess the request
        processed_request = self._preprocess_request(request, **kwargs)
        # 如果启用文档分析,在后台启动(不等待)
        if request.enable_analyze_document:
            asyncio.create_task(self._analyze_document_async(processed_request))
        # Delegate to the actual implementation
        # 主要任务:构建索引(等待完成)
        result = await asyncio.to_thread(self.execute, processed_request, **kwargs)
        return result

    @staticmethod
    def _setup_metadata(request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        """
        Setup metadata for the request.
        Override this method in subclasses to add specific metadata setup logic.
        """
        metadata = request.metadata if request.metadata else {}
        if metadata.get(MetadataConstants.FILENAME) is None:
            metadata.update({MetadataConstants.FILENAME: get_file_name_from_path(request.save_to_path)})
        if metadata.get(MetadataConstants.SOURCE_TYPE, None) is None:
            metadata.update({MetadataConstants.SOURCE_TYPE: SourceTypeEnum.UPLOAD_FILES.value})
        request.metadata = metadata
        return metadata

    @abstractmethod
    def execute(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        raise NotImplementedError

    def _physical_delete_task_sql_by_user_folder_id(self, params=None):
        if params is None:
            params = {}
        delete_sql = """
            delete from build_knowledge_index_task 
            where knowledge_index_folder_id = %s
        """
        if params.get("knowledge_index_folder_id", None) is None:
            logger.error(f"[_delete_task_sqll_by_user_folder_id]-id is required")
            raise ValueError(f"[_delete_task_sqll_by_user_folder_id]-id is required")
        delete_params = (params.get("knowledge_index_folder_id"),)
        if self.cursor is None:
            return
        try:
            self.cursor.execute(delete_sql, delete_params)
            self.cursor.connection.commit()
            logger.info(f"Executing SQL: {delete_sql} with params: {delete_params}")
            logger.info(
                f"[_delete_task_sqll_by_user_folder_id]-Delete id:{params.get('knowledge_index_folder_id')} successfully.")
        except Exception as e:
            logger.exception(f"[_delete_task_sqll_by_user_folder_id]-Error occurred when delete task: {e}")
            raise Exception(f"[_delete_task_sqll_by_user_folder_id]-Error occurred when delete task: {e}")

    def _do_update_task_status_sql_by_user_folder_id(self, params=None):
        if params is None:
            params = {}
        update_sql = """
            update build_knowledge_index_task 
            set status = %s, err_info = %s, docs_index_count = %s, child_docs_index_count = %s, update_at = %s
            where knowledge_index_folder_id = %s
        """
        if params.get("status", None) is None:
            logger.error(f"[_do_update_task_status_sql]-status is required")
            raise ValueError(f"[update_status]-status is required")
        if params.get("knowledge_index_folder_id", None) is None:
            logger.error(f"[_do_update_task_status_sql]-id is required")
            raise ValueError(f"[_do_update_task_status_sql]-id is required")
        update_params = (params.get("status"), params.get("err_info", None),
                         params.get("docs_index_count", None),
                         params.get("child_docs_index_count", None),
                         get_beijing_current_time(), params.get("knowledge_index_folder_id"))
        if self.cursor is None:
            return
        try:
            self.cursor.execute(update_sql, update_params)
            self.cursor.connection.commit()
            logger.info(f"Executing SQL: {update_sql} with params: {update_params}")
            logger.info(
                f"[_do_update_task_status_sql]-Update id:{params.get('knowledge_index_folder_id')} status to: {params.get('status')} successfully.")
        except Exception as e:
            logger.exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")
            raise Exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")

    def _do_update_task_status_sql_by_user_folder_name(self, params):
        if params is None:
            params = {}
        update_sql = """
            update build_knowledge_index_task 
            set status = %s, err_info = %s, docs_index_count = %s, child_docs_index_count = %s, update_at = %s
            where knowledge_index_folder_name = %s
        """
        if params.get("status", None) is None:
            logger.error(f"[_do_update_task_status_sql]-status is required")
            raise ValueError(f"[update_status]-status is required")
        if params.get("knowledge_index_folder_name", None) is None:
            logger.error(f"[_do_update_task_status_sql]-knowledge_index_folder_name is required")
            raise ValueError(f"[_do_update_task_status_sql]-knowledge_index_folder_name is required")
        update_params = (params.get("status"), params.get("err_info", None),
                         params.get("docs_index_count", None),
                         params.get("child_docs_index_count", None),
                         get_beijing_current_time(), params.get("knowledge_index_folder_name"))
        if self.cursor is None:
            return
        try:
            self.cursor.execute(update_sql, update_params)
            self.cursor.connection.commit()
            logger.info(f"Executing SQL: {update_sql} with params: {update_params}")
            logger.info(
                f"[_do_update_task_status_sql]-Update id:{params.get('knowledge_index_folder_id')} status to: {params.get('status')} successfully.")
        except Exception as e:
            logger.exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")
            raise Exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")

    def _do_update_task_status_sql(self, params=None):
        if params is None:
            params = {}
        update_sql = """
            update build_knowledge_index_task 
            set status = %s, err_info = %s, docs_index_count = %s, child_docs_index_count = %s, update_at = %s
            where id = %s
        """
        if params.get("status", None) is None:
            logger.error(f"[_do_update_task_status_sql]-status is required")
            raise ValueError(f"[update_status]-status is required")
        if params.get("id", None) is None:
            logger.error(f"[_do_update_task_status_sql]-id is required")
            raise ValueError(f"[_do_update_task_status_sql]-id is required")
        update_params = (params.get("status"), params.get("err_info", None),
                         params.get("docs_index_count", None),
                         params.get("child_docs_index_count", None),
                         get_beijing_current_time(), params.get("id"))
        if self.cursor is None:
            return
        try:
            self.cursor.execute(update_sql, update_params)
            self.cursor.connection.commit()
            logger.info(f"Executing SQL: {update_sql} with params: {update_params}")
            logger.info(
                f"[_do_update_task_status_sql]-Update id:{params.get('id')} status to: {params.get('status')} successfully.")
        except Exception as e:
            logger.exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")
            raise Exception(f"[_do_update_task_status_sql]-Error occurred when update status: {e}")

    def del_ori_file(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            if request.save_to_path is not None and os.path.exists(request.save_to_path):
                os.remove(request.save_to_path)
        finally:
            params = {
                "status": BuildKnowledgeIndexStatusEnum.DELETE.value,  # 删除
                "id": request.id,
                "err_info": None,
            }
            try:
                self._do_update_task_status_sql(params)
                self.handler.release()
            except Exception as e:
                logger.exception(f"[del_some_indies_in_local_vs]-Error occurred when update status in mysql:")
                raise Exception(f"[del_some_indies_in_local_vs]-Error occurred when update status in mysql: {e}")

    def retry_failed_task(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        pass

    @abstractmethod
    def del_part_of_indies(self, request: DeleteKnowledgeBaseIndexRequestVO):
        """
        删除某个版本的部分索引
        """
        raise NotImplementedError

    @abstractmethod
    def del_version_of_indies_and_docs(self, request: DeleteKnowledgeBaseIndexRequestVO):
        """
        删除某个版本的所有索引和以及storage保存的文档
        """
        raise NotImplementedError

    @abstractmethod
    def del_workspace(self, request: DeleteKnowledgeBaseIndexRequestVO):
        """
        删除所有版本的所有索引和以及storage保存的文档
        """
        raise NotImplementedError

    def close_mysql_client(self):
        # self.mysql_client.close()
        pass

    def _setup_workdone_fields(self, request):
        return {
            "status": BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value,  # 执行成功
            "id": request.id,
            "ori_file_path": request.save_to_path,
            "err_info": None
        }

    def _setup_delete_indies_fields(self, request):
        docs_index_count = request.docs_index_count
        child_docs_index_count = request.child_docs_index_count

        params = {
            "status": BuildKnowledgeIndexStatusEnum.DELETE.value,  # 删除
            "id": request.id,
            "err_info": None,
            "docs_index_count": docs_index_count,
            "child_docs_index_count": child_docs_index_count
        }
        return params

    def _handle_exception(self, e, params):
        if isinstance(e, UnicodeDecodeError):
            logger.exception(f"[execute]-Error occurred when making vector store with UnicodeDecodeError")
            status = BuildKnowledgeIndexStatusEnum.FILE_DECODING_FAILED.value
            err_info = str(e)
        elif isinstance(e, AssertionError):
            logger.exception(
                f"[execute]-Error occurred when making vector store with AssertionError, It seems missing some required request parameters")
            status = BuildKnowledgeIndexStatusEnum.MISSING_REQUIRED_PARAMETERS.value
            err_info = str(e)
        elif isinstance(e, RetryableStepFailedException):
            status = e.status
            err_info = str(e)
        else:
            logger.exception(f"[execute]-Error occurred when making vector store:{e}")
            status = BuildKnowledgeIndexStatusEnum.UNKNOWN_ERROR.value
            err_info = str(e)
        params.update({"status": status, "err_info": err_info})

    def _finalize(self, params):
        try:
            self._do_update_task_status_sql(params)
            # self.handler.release()
        except Exception as e:
            logger.exception(f"[del_some_indies_in_local_vs]-Error occurred when update status in mysql:")
            raise Exception(f"[del_some_indies_in_local_vs]-Error occurred when update status in mysql: {e}")

    def _finalize_delete_version_of_tasks(self, knowledge_index_folder_name):
        params = {
            "status": BuildKnowledgeIndexStatusEnum.DELETE.value,  # 删除
            "knowledge_index_folder_name": knowledge_index_folder_name,
            "docs_index_count": 0,
            "child_docs_index_count": 0
        }
        try:
            self._do_update_task_status_sql_by_user_folder_name(params)
        except Exception as e:
            logger.exception(f"[del_all_indies_and_docs]-Error occurred when update status in mysql:")

    def _finalize_delete_all(self, knowledge_index_folder_id):
        params = {
            "status": BuildKnowledgeIndexStatusEnum.DELETE.value,  # 删除
            "knowledge_index_folder_id": knowledge_index_folder_id,
            "docs_index_count": 0,
            "child_docs_index_count": 0
        }
        try:
            self._physical_delete_task_sql_by_user_folder_id(params)
        except Exception as e:
            logger.exception(f"[del_all_indies_and_docs]-Error occurred when update status in mysql:")

    def _finalize_execution(self, params: dict,
                            request: BuildKnowledgeIndexItemRequestVO) -> BuildKnowledgeIndexItemRequestVO:
        if self.handler:
            docs_index_count, child_docs_index_count = self.handler.get_docs_index_count()
            if docs_index_count is not None:
                params.update({"docs_index_count": docs_index_count,
                               "child_docs_index_count": child_docs_index_count
                               })
            self._finalize(params)
        request.build_status = params.get("status")
        return request

    def _load_retryable_info(self, request, params, steps_enum):
        retryable_context_file = os.path.join(MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR,
                                              request.knowledge_index_folder_id,
                                              CREATE_INDEX_RETRYABLE_CONTEXT_FILE_DIR_NAME,
                                              f"{request.id}_retryable_context.json")
        if not os.path.exists(retryable_context_file):
            logger.warning(f"[retry_failed_task]-Retryable context file not exists: {retryable_context_file}")
            params.update({"status": BuildKnowledgeIndexStatusEnum.RETRY_REQUIRED_FILE_NOT_EXISTS.value,
                           "err_info": "Retryable context file not exists"})
            raise FileNotFoundError(
                f"[retry_failed_task]-Retryable context file not exists: {retryable_context_file}")
        with open(retryable_context_file, "r") as f:
            failed_task_info = json.load(f)

        retryable_context_info_filepath = failed_task_info.get("retryable_context_info_filepath")
        retryable_context_info = {}
        if not os.path.exists(retryable_context_info_filepath):
            logger.info(
                f"[retry_failed_task]-Retryable context info file not exists: {retryable_context_info_filepath}, so just init a new one.")
        else:
            with open(retryable_context_info_filepath, "rb") as f:
                retryable_context_info = pickle.load(f)

        retryable_request_params = failed_task_info.get("request_params", {})
        failed_task_info.update(
            {"id": request.id, "is_triggered_by_retry_task": True, "user_folder_id": request.knowledge_index_folder_id})
        retryable_context = CreateIndexRetryableContext(**failed_task_info,
                                                        steps_enum=steps_enum)
        return retryable_context, retryable_context_info, retryable_request_params


class BuildingHippoRAGGraphIndexStrategy(BaseBuildingIndexStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_db_prefix = GRAPH_DB_PREFIX

    def execute(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        logger.info(f"[execute]-Start build knowledge index with request: {request.model_dump()}")
        params = self._setup_workdone_fields(request)
        try:
            document_id_prefix = request.id
            neo4j_graph_db_name = self.graph_db_prefix + request.knowledge_index_folder_id + "ver" + str(request.checkout_version)  # 多版本控制
            fp = detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8")
            kg_index_product_id = request.build_kg_index_product_id
            if not kg_index_product_id:
                kg_index_product_id = SupportedModelsInfoUtil.default_build_kg_index_product_id()
                assert kg_index_product_id is not None, "Build knowledge index product id is None, please provide a valid one!"
            self.handler = DefaultBuildingHippoRAGGraphHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=fp,
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                neo4j_graph_db_name=neo4j_graph_db_name,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config)
            metadata = self._setup_metadata(request, **kwargs)
            self.handler.precompute_and_save_results(
                user_id=request.user_id,
                product_id=kg_index_product_id,
                metadata=metadata)
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            return self._finalize_execution(params, request)

    def del_part_of_indies(self, request: DeleteKnowledgeBaseIndexRequestVO):
        self.handler = DefaultBuildingHippoRAGGraphHandler(
            id=request.id,
            checkout_version=request.checkout_version,
            local_doc_folder_name=request.knowledge_index_folder_name,
            is_chinese_text=request.is_chinese_text,
            neo4j_graph_db_name=self.graph_db_prefix + request.knowledge_index_folder_id + "ver" + str(request.checkout_version),
            # 多版本控制,
            user_folder_id=request.knowledge_index_folder_id
        )
        docs_index_count = request.docs_index_count

        params = self._setup_delete_indies_fields(request)
        if docs_index_count is None:
            return
        try:
            doc_index_id_prefix = request.id
            _, ids = generate_document_ids(document_id_prefix=doc_index_id_prefix, num_docs=request.docs_index_count)
            self.handler.del_part_of_indies(ids)
            # params.update({"docs_index_count": 0})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)

    def del_version_of_indies_and_docs(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingHippoRAGGraphHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text,
                neo4j_graph_db_name=self.graph_db_prefix + request.knowledge_index_folder_id + "ver" + str(request.checkout_version),
                # 多版本控制,
                user_folder_id=request.knowledge_index_folder_id,
            )
            self.handler.del_version_of_indies_and_docs()
        except Exception as e:
            logger.exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store:")
            raise Exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store {e}")
        finally:
            self._finalize_delete_version_of_tasks(request.knowledge_index_folder_name)

    def del_workspace(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingHippoRAGGraphHandler(
                id=request.id,
                user_folder_id=request.knowledge_index_folder_id,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text,
                neo4j_graph_db_name=self.graph_db_prefix + request.knowledge_index_folder_id + "ver" + str(
                    request.checkout_version),
                # 多版本控制,
            )
            self.handler.del_workspace()
        except Exception as e:
            logger.exception(f"[del_workspace]-Error occurred when delete workspace:")
            raise Exception(f"[del_workspace]-Error occurred when delete workspace {e}")
        finally:
            self._finalize_delete_all(request.knowledge_index_folder_id)

    def retry_failed_task(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        params = {
            "status": BuildKnowledgeIndexStatusEnum.INIT_STATUS.value,
            "id": request.id,
            "err_info": None
        }
        try:
            retryable_context, retryable_context_info, retryable_request_params = self._load_retryable_info(request,
                                                                                                            params,
                                                                                                            CreateHippoRAGStepsEnum)
            self.handler = DefaultBuildingHippoRAGGraphHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8"),
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                neo4j_graph_db_name=self.graph_db_prefix + request.knowledge_index_folder_id + "ver" + str(request.checkout_version),
                # 多版本控制,
                retryable_context=retryable_context,
                retryable_context_info=retryable_context_info,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config,
            )
            self.handler.precompute_and_save_results(**retryable_request_params)
            docs_index_count, child_docs_index_count = self.handler.get_docs_index_count()
            params.update({"docs_index_count": docs_index_count,
                           "child_docs_index_count": child_docs_index_count,
                           "status": BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)


class BuildingVectorStoreIndexStrategy(BaseBuildingIndexStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        logger.info(f"[execute]-Start making vector store with request: {request.model_dump()} by user_id: {request.user_id}")
        params = self._setup_workdone_fields(request)
        try:
            fp = detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8")
            self.handler = DefaultBuildingVectorStoreHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=fp,
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config)
            metadata = self._setup_metadata(request, **kwargs)
            self.handler.precompute_and_save_results(metadata=metadata)
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            return self._finalize_execution(params, request)

    def del_part_of_indies(self, request: DeleteKnowledgeBaseIndexRequestVO):
        self.handler = DefaultBuildingVectorStoreHandler(
            checkout_version=request.checkout_version,
            user_folder_id=request.knowledge_index_folder_id,
            local_doc_folder_name=request.knowledge_index_folder_name,
            is_chinese_text=request.is_chinese_text)
        docs_index_count = request.docs_index_count
        if docs_index_count is None or docs_index_count <= 0:
            return

        params = self._setup_delete_indies_fields(request)
        try:
            doc_index_id_prefix = request.id
            _, ids = generate_document_ids(document_id_prefix=doc_index_id_prefix, num_docs=request.docs_index_count)
            _, sub_ids = generate_sub_document_ids(document_id_prefix=doc_index_id_prefix,
                                                   num_docs=request.child_docs_index_count)
            self.handler.del_part_of_indies(ids, sub_ids=sub_ids)
            # params.update({"docs_index_count": 0})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)

    def del_version_of_indies_and_docs(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingVectorStoreHandler(
                checkout_version=request.checkout_version,
                user_folder_id=request.knowledge_index_folder_id,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text, )
            self.handler.del_version_of_indies_and_docs()
        except Exception as e:
            logger.exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store:{e}")
            raise Exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store {e}")
        finally:
            self._finalize_delete_version_of_tasks(request.knowledge_index_folder_name)

    def del_workspace(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingVectorStoreHandler(
                checkout_version=request.checkout_version,
                user_folder_id=request.knowledge_index_folder_id,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text, )
            self.handler.del_workspace()
        except Exception as e:
            logger.exception(f"[del_workspace]-Error occurred when delete workspace:{e}")
            raise Exception(f"[del_workspace]-Error occurred when delete workspace {e}")
        finally:
            self._finalize_delete_all(request.knowledge_index_folder_id)

    def retry_failed_task(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        logger.info(f"[retry_failed_task]-Start retry failed task with request: {request.model_dump()}")
        params = {
            "status": BuildKnowledgeIndexStatusEnum.INIT_STATUS.value,
            "id": request.id,
            "err_info": None
        }
        try:
            retryable_context, retryable_context_info, retryable_request_params = self._load_retryable_info(request,
                                                                                                            params,
                                                                                                            CreateVectorStoreIndexStepsEnum)
            self.handler = DefaultBuildingVectorStoreHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8"),
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                retryable_context=retryable_context,
                retryable_context_info=retryable_context_info,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config,
            )
            self.handler.precompute_and_save_results(**retryable_request_params)
            docs_index_count, child_docs_index_count = self.handler.get_docs_index_count()
            params.update({"docs_index_count": docs_index_count,
                           "child_docs_index_count": child_docs_index_count,
                           "status": BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)

    def housekeeping_old_knowledge_version(self, relative_knowledge_folder_name: str):
        # 删除所有以relative_knowledge_folder_name开头的文件夹
        dirs = glob.glob(os.path.join(KB_ROOT_PATH, f"{relative_knowledge_folder_name}*"))
        for dir in dirs:
            try:
                shutil.rmtree(dir)
                logger.info(f"[housekeeping_old_knowledge_version]-Delete old knowledge version folder: {dir}")
            except Exception as e:
                logger.exception(
                    f"[housekeeping_old_knowledge_version]-Error occurred when delete old knowledge version folder: {e}")


class BuildingElasticSearchIndexStrategy(BaseBuildingIndexStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs) -> BuildKnowledgeIndexItemRequestVO:
        logger.info(f"[execute]-Start making vector store with request: {request.model_dump()}")
        params = self._setup_workdone_fields(request)
        try:
            fp = detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8")
            self.handler = DefaultBuildingElasticSearchIndexHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=fp,
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config)
            metadata = self._setup_metadata(request, **kwargs)
            self.handler.precompute_and_save_results(metadata=metadata)
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            return self._finalize_execution(params, request)

    def del_part_of_indies(self, request: DeleteKnowledgeBaseIndexRequestVO):
        self.handler = DefaultBuildingElasticSearchIndexHandler(
            id=request.id,
            checkout_version=request.checkout_version,
            user_folder_id=request.knowledge_index_folder_id,
            local_doc_folder_name=request.knowledge_index_folder_name,
            is_chinese_text=request.is_chinese_text)
        docs_index_count = request.docs_index_count
        if docs_index_count is None or docs_index_count <= 0:
            return

        params = self._setup_delete_indies_fields(request)
        try:
            doc_index_id_prefix = request.id
            _, ids = generate_document_ids(document_id_prefix=doc_index_id_prefix, num_docs=request.docs_index_count)
            self.handler.del_part_of_indies(ids)
            # params.update({"docs_index_count": 0})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)

    def del_version_of_indies_and_docs(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingElasticSearchIndexHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                user_folder_id=request.knowledge_index_folder_id,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text, )
            self.handler.del_version_of_indies_and_docs(index_name=request.knowledge_index_folder_id)
        except Exception as e:
            logger.exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store:")
            raise Exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store {e}")
        finally:
            self._finalize_delete_version_of_tasks(request.knowledge_index_folder_name)

    def del_workspace(self, request: DeleteKnowledgeBaseIndexRequestVO):
        try:
            self.handler = DefaultBuildingElasticSearchIndexHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                user_folder_id=request.knowledge_index_folder_id,
                local_doc_folder_name=request.knowledge_index_folder_name,
                is_chinese_text=request.is_chinese_text, )
            self.handler.del_workspace()
        except Exception as e:
            logger.exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store:")
            raise Exception(f"[del_all_local_vs_dir]-Error occurred when delete all local vector store {e}")
        finally:
            self._finalize_delete_all(request.knowledge_index_folder_id)

    def retry_failed_task(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        params = {
            "status": BuildKnowledgeIndexStatusEnum.INIT_STATUS.value,
            "id": request.id,
            "err_info": None
        }
        try:
            retryable_context, retryable_context_info, retryable_request_params = self._load_retryable_info(request,
                                                                                                            params,
                                                                                                            CreateElasticSearchIndexStepsEnum)
            self.handler = DefaultBuildingElasticSearchIndexHandler(
                id=request.id,
                checkout_version=request.checkout_version,
                local_doc_folder_name=request.knowledge_index_folder_name,
                ingest_file_path=detect_encoding_and_convert_file(request.save_to_path, encoding_name="utf-8"),
                is_chinese_text=request.is_chinese_text,
                document_id_prefix=request.id,
                retryable_context=retryable_context,
                retryable_context_info=retryable_context_info,
                user_folder_id=request.knowledge_index_folder_id,
                **request.build_knowledge_index_config,
            )
            self.handler.precompute_and_save_results(**retryable_request_params)
            docs_index_count, child_docs_index_count = self.handler.get_docs_index_count()
            params.update({"docs_index_count": docs_index_count,
                           "child_docs_index_count": child_docs_index_count,
                           "status": BuildKnowledgeIndexStatusEnum.WORK_DONE_SUCCESS.value})
        except Exception as e:
            self._handle_exception(e, params)
        finally:
            self._finalize(params)

