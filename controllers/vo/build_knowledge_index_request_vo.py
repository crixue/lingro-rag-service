from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.enums.raw_file_type_enum import RawFileTypeEnum


class SerializePromptRequestVO(BaseModel):
    title: str = Field(default=None)
    prompt_folder_name: str = Field(default=None, required=True)
    serialize_info: dict = Field(default={})


DEFAULT_MAKE_VS_CONFIG = {
    # "cnChunkSize": 500,
    # "cnChunkOverlap": 0,
    # "enChunkSize": 800,
    # "enChunkOverlap": 0
}

class BuildKnowledgeIndexItemRequestVO(BaseModel):
    """
    make_vs_config contains:
      - "mode": "single" or
      - "chunking_strategy": "basic"
      - "max_characters": int
      - "preferred_max_characters": int
      - "overlap": int
      And if loaded file is json or jsonl, it also contains:
      - "jq_schema": str
      - "json_lines": bool
      - "text_content": bool
      And if retrieve type is local vector store, it also contains:
      - "distance_strategy": "EUCLIDEAN_DISTANCE" or "MAX_INNER_PRODUCT", default is "EUCLIDEAN_DISTANCE"
      - "normalize_L2": bool, default is False
    """
    id: str = Field(default=None)
    build_knowledge_index_type: BuildKnowledgeIndexTypeEnum = Field(default=None, alias="buildKnowledgeIndexType")
    raw_file_type: RawFileTypeEnum = Field(default="DEFAULT",
                                           description="DEFAULT, CONTENT_STRING, URL, SPEECH_RECOGNIZE_TRANSCRIPT, OCR_PARSE_TRANSCIPT",
                                           alias="rawFileType")
    save_to_path: str = Field(default=None, alias="saveToPath")
    is_chinese_text: Optional[bool] = Field(default=True, alias="isChineseText")
    knowledge_index_folder_id: str = Field(default=None, description="user_folder id", alias="knowledgeIndexFolderId")
    knowledge_index_folder_name: Optional[str] = Field(default=None, alias="knowledgeIndexFolderName")
    user_id: str = Field(default=None, description="user id", alias="triggerUserId")
    build_kg_index_product_id: Optional[str] = Field(default=None, description="build knowledge index llm product id")
    build_knowledge_index_config: Optional[dict] = Field(default=DEFAULT_MAKE_VS_CONFIG, alias="buildKnowledgeIndexConfig")
    is_triggered_by_retry_task: bool = Field(default=False, alias="isTriggeredByRetryTask")
    metadata: Optional[dict] = Field(default=None, alias="metadata")
    build_status: Optional[str] = Field(default=None, alias="buildStatus")
    checkout_version: int = Field(default=None, alias="checkoutVersion")
    enable_analyze_document: bool = Field(default=False, alias="enableAnalyzeDocument", description="whether to enable document analysis when building knowledge index")
    token_limits: int = Field(default=128000, alias="tokenLimits", description="The token limits for building knowledge index")

    @field_validator("knowledge_index_folder_id")
    def validate_knowledge_index_folder_id(cls, field):
        if field is None:
            raise ValueError(f"knowledgeIndexFolderId should not be None")
        return field

    @field_validator("knowledge_index_folder_name")
    def validate_knowledge_index_folder_name(cls, field):
        if field is None:
            raise ValueError(f"knowledgeIndexFolderName should not be None")
        return field

    @field_validator("id")
    def validate_id(cls, field):
        if field is None:
            raise ValueError(f"id should not be None")
        return field

    @field_validator("build_knowledge_index_type")
    def validate_make_tool(cls, field):
        if field is None:
            raise ValueError(f"build_knowledge_index_type should not be None")
        return field

    @field_validator("save_to_path")
    def validate_ori_file_path(cls, field):
        if field is None:
            raise ValueError(f"save_to_path should not be None")
        return field

    @field_validator("build_knowledge_index_config", mode="before")
    def set_default_build_knowledge_index_config(cls, field):
        if field is None:
            return DEFAULT_MAKE_VS_CONFIG
        return field

    @model_validator(mode="after")
    def validate_after(self):
        build_knowledge_index_config = self.build_knowledge_index_config
        if build_knowledge_index_config is not None:
            self.build_kg_index_product_id = build_knowledge_index_config.get("build_kg_index_product_id", None)
        return self


class DeleteKnowledgeBaseIndexRequestVO(BaseModel):
    id: str = Field(default=None)
    build_knowledge_index_type: BuildKnowledgeIndexTypeEnum = Field(default=None, alias="buildKnowledgeIndexType")
    save_to_path: Optional[str] = Field(default=None, alias="saveToPath")
    knowledge_index_folder_id: str = Field(default=None, description="user_folder id", alias="knowledgeIndexFolderId")
    knowledge_index_folder_name: Optional[str] = Field(default=None, alias="knowledgeIndexFolderName")
    is_chinese_text: Optional[bool] = Field(default=True, alias="isChineseText")
    docs_index_count: Optional[int] = Field(default=None, alias="docsIndexCount")
    child_docs_index_count: Optional[int] = Field(default=None, alias="childDocsIndexCount")
    checkout_version: Optional[int] = Field(default=None, alias="checkoutVersion")

    @field_validator("build_knowledge_index_type")
    def validate_make_tool(cls, field):
        if field is None:
            raise ValueError(f"buildKnowledgeIndexType should not be None")
        return field

    @field_validator("knowledge_index_folder_id")
    def validate_user_folder_id(cls, field):
        if field is None:
            raise ValueError(f"knowledgeIndexFolderId should not be None")
        return field

    @field_validator("id")
    def validate_id(cls, field):
        if field is None:
            raise ValueError(f"id should not be None")
        return field

    @field_validator("knowledge_index_folder_name")
    def validate_local_vs_folder_name(cls, field):
        if field is None:
            raise ValueError(f"knowledgeIndexFolderName should not be None")
        return field

    @field_validator("is_chinese_text")
    def validate_is_chinese_text(cls, field):
        if field is None:
            return True
        return field


class BuildKnowledgeIndexBundleRequestVO(BaseModel):
    build_knowledge_index_item_request_list: list[BuildKnowledgeIndexItemRequestVO] = Field(default=None, alias="buildKnowledgeIndexItemRequestVOList")
    knowledge_index_version: str = Field(default=None, alias="knowledgeIndexVersion")
    knowledge_index_mv_control_id: str = Field(default=None, alias="knowledgeIndexMVControlId")
    build_knowledge_index_type: BuildKnowledgeIndexTypeEnum = Field(default=None, alias="buildKnowledgeIndexType")
    knowledge_index_folder_id: str = Field(default=None, description="user_folder id", alias="knowledgeIndexFolderId")
    knowledge_index_folder_name: Optional[str] = Field(default=None, alias="knowledgeIndexFolderName")
    trigger_user_id: str = Field(default=None, alias="triggerUserId")
    require_to_delete_version_requests: Optional[list[DeleteKnowledgeBaseIndexRequestVO]] = Field(default=None, alias="requireToDeleteVersionRequests")