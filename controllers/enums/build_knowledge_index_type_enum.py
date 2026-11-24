from enum import Enum


class BuildKnowledgeIndexTypeEnum(Enum):
    LOCAL_VECTOR_STORE = "A01"
    ELASTICSEARCH = "A02"
    HIPPO_KNOWLEDGE_GRAPH = "A03"
    WEB_SEARCH = "B01"
    MIXED_LOCAL_VECTOR_STORE_AND_WEB_SEARCH = "A01B01"
    MIXED_ELASTICSEARCH_AND_WEB_SEARCH = "A02B01"
    MIXED_HIPPO_KNOWLEDGE_GRAPH_AND_WEB_SEARCH = "A03B01"

    @staticmethod
    def values():
        return [e.value for e in BuildKnowledgeIndexTypeEnum]
