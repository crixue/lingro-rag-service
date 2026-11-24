
DOC_STORE_SUFFIX = "_doc_store"

class MetadataConstants:
    """元数据键常量类，用于统一管理文档元数据的键名"""
    # 基本信息
    SOURCE = "source"
    FILENAME = "filename"
    URL = "url"
    TITLE = "title"
    AUTHOR = "author"
    CREATED_AT = "created_at"
    IS_CHINESE_TEXT = "is_chinese_text"

    # 版本和ID相关
    CHECKOUT_VERSION = "checkout_version"
    KNOWLEDGE_FOLDER_ID = "knowledge_folder_id"

    # 来源相关
    SOURCE_TYPE = "source_type"

    #score
    BM25_SCORE = "bm25_score"
    FAISS_SCORE = "faiss_score"
    HIPPO_RAG_SCORE = "hippo_rag_score"
    SCORE = "score"

    # 其他元数据字段
    DOCUMENT_ID = "document_id"
    DOCUMENT_TYPE = "document_type"
    LAST_UPDATED_AT = "last_updated_at"
    DOC_ID = "doc_id"
    PREFIX_ID = "prefix_id"
    SEQ_NUM = "seq_num"

    # 分析文档的元数据字段
    ANALYSIS_GLOBAL_CORE_SUMMARY = "global_core_summary"
    ANALYSIS_GLOBAL_THEME_TAGS = "global_theme_tags"
    ANALYSIS_GLOBAL_DOCUMENT_INTENT_TYPES = "global_document_intent_types"
    ANALYSIS_GLOBAL_KEY_ENTITIES = "global_key_entities"

    @staticmethod
    def popup_useless_metadata(metadata: dict, other_useless_keys: list[str] = None) -> dict:
        """
        清除无用的元数据字段
        :param metadata: dict, 元数据字典
        :return: dict, 清理后的元数据字典
        """
        if not metadata:
            return {}
        useless_keys = [
            MetadataConstants.SOURCE,
            MetadataConstants.IS_CHINESE_TEXT,
            MetadataConstants.CHECKOUT_VERSION,
            MetadataConstants.KNOWLEDGE_FOLDER_ID,
            MetadataConstants.DOCUMENT_ID,
            MetadataConstants.DOCUMENT_TYPE,
            MetadataConstants.LAST_UPDATED_AT,
            MetadataConstants.DOC_ID,
            MetadataConstants.PREFIX_ID,
            MetadataConstants.SEQ_NUM,
            MetadataConstants.BM25_SCORE,
            MetadataConstants.FAISS_SCORE,
            MetadataConstants.HIPPO_RAG_SCORE,
        ]
        if other_useless_keys:
            useless_keys.extend(other_useless_keys)

        return {k: v for k, v in metadata.items() if k not in useless_keys}
