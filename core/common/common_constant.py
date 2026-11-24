TOTAL_TOKENS_KEY = "total_tokens"
USER_ID_KEY = "user_id"
PRODUCT_ID_KEY = "product_id"
REQUEST_ID_KEY = "request_id"

SEARCH_RELEVANT_DOCS_TOOL_NAME = "search_relevant_docs"

FALL_BACK_ANSWER = "Sorry, I don't know the answer"
FALL_BACK_ANSWER_ZH = "对不起，我不知道答案"

RATE_LIMIT_ERROR_MESSAGE = "This model is rate-limited, please switch to another model or try again later."
RATE_LIMIT_ERROR_MESSAGE_ZH = "该模型已被限速，请切换其他模型或稍后再试。"

PLACEHOLDER_STRING = "<PLACEHOLDER>"

# ref: prompt_definitions/builtin/lang/cn/analysis_document_system_prompt.json
class DocumentAnalysisConstants:
    CORE_SUMMARY = "core_summary"
    THEME_TAGS = "theme_tags"
    DOCUMENT_INTENT_TYPES = "document_intent_type"
    KEY_ENTITIES = "key_entities"
    MAP_PHASE_SUMMARY = "map_phase_summary"