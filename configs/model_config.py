import logging
import os
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Running environment: local / dev / prod
RUNNING_ENV = os.getenv("APP_ENV", "local")
# 本地开发默认只告警；生产环境（APP_ENV != local）强制校验，也可用 STRICT_ENV_VALIDATION=true 强制开启
STRICT_ENV_VALIDATION = os.getenv("STRICT_ENV_VALIDATION", "").lower() in {"1", "true", "yes"}

# 定义所有必需的环境变量(没有默认值的配置)
REQUIRED_ENV_VARS = [
    "REMOTE_MIDDLEWARE_HOST",
    "MYSQL_DBNAME",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "ELASTICSEARCH_URL",
    "MICRO_LLM_BACKEND_CORE_BASE_URL",
    "MICRO_LLM_BACKEND_WEBSEARCH_ENGINE_BASE_URL",
    "MICRO_LLM_BACKEND_PAYMENT_ORDER_BASE_URL",
    "MICRO_LLM_BACKEND_USER_BASE_URL",
    "QWEN_SERIES_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GLM_4_PLUS_API_KEY",
    "LINGRO_LANGCHAIN_PY_CLIENT_SECRET",
]


def validate_required_env_vars():
    """验证所有必需的环境变量是否已设置"""
    missing_vars = []

    for var_name in REQUIRED_ENV_VARS:
        value = os.environ.get(var_name)
        if value is None or value.strip() == "":
            missing_vars.append(var_name)

    if missing_vars:
        error_message = (
            f"Missing Required Vars in Environment:\n"
            f"{', '.join(missing_vars)}\n"
        )
        if RUNNING_ENV == "local" and not STRICT_ENV_VALIDATION:
            # 本地开发允许先启动服务：模型调用、检索、计费会因缺少配置而失败，需补齐 .env 后才能联调
            logging.getLogger(__name__).warning(
                "Non-strict mode (APP_ENV=local): missing environment variables, "
                "the service will start but dependent features will fail.\n%s", error_message
            )
            return
        raise EnvironmentError(error_message)


# 在配置加载前验证环境变量
validate_required_env_vars()

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
# Set default logging level to WARNING
logging.getLogger().setLevel(logging.WARNING)
# Set specific logging level for the current project to INFO
logging.getLogger(__name__).setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=LOG_FORMAT)
# 日志目录可通过 LOG_FILE_DIR 覆盖；Windows 下默认落到项目内 logs/，避免写入系统盘根目录
LOG_FILE_DIR = os.getenv("LOG_FILE_DIR", "/tmp/logs" if os.name != "nt" else os.path.join(BASE_DIR, "logs"))
try:
    os.makedirs(LOG_FILE_DIR, exist_ok=True)
except OSError:
    LOG_FILE_DIR = os.path.join(BASE_DIR, "logs")
    os.makedirs(LOG_FILE_DIR, exist_ok=True)
handler = TimedRotatingFileHandler(os.path.join(LOG_FILE_DIR, "langchain-local-llm.log"), when='midnight', interval=1,
                                   backupCount=7)
handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(handler)

logger.info(f"App Start... Current running environment: {RUNNING_ENV}")

# application configs，"/llm-backend" 路径会单独处理，不受下面 protected_paths 影响
protected_paths = [
    "/knowledge-index",
]

# Basic Configuration
DEFAULT_CONTEXT_TOKENS_LIMIT = int(os.getenv("DEFAULT_CONTEXT_TOKENS_LIMIT", 6000))
VS_ROOT_PATH = os.getenv("VS_ROOT_PATH", "/vector_store")
CACHE_ROOT_PATH = os.getenv("CACHE_ROOT_PATH", "/temp/cache")
BASE_PROMPT_TEMPLATE_PATH = os.getenv("BASE_PROMPT_TEMPLATE_PATH", "/prompt_templates")

# Remote Middleware Host
REMOTE_MIDDLEWARE_HOST = os.getenv("REMOTE_MIDDLEWARE_HOST")

# MySQL Configuration
MYSQL_CONFIG = {
    "MYSQL_HOST": os.getenv("MYSQL_HOST", REMOTE_MIDDLEWARE_HOST),
    "MYSQL_PORT": int(os.getenv("MYSQL_PORT", 3306)),
    "MYSQL_DBNAME": os.getenv("MYSQL_DBNAME"),
    "MYSQL_USER": os.getenv("MYSQL_USER"),
    "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
    "MYSQL_CHARSET": "utf8mb4"
}

# Redis Configuration
REDIS_CONFIG = {
    "REDIS_HOST": os.getenv("REDIS_HOST", REMOTE_MIDDLEWARE_HOST),
    "REDIS_PORT": int(os.getenv("REDIS_PORT", 6379)),
    "REDIS_DB": int(os.getenv("REDIS_DB", 0)),
}

# Neo4j Configuration
NEO4J_CONFIG = {
    "NEO4J_URI": os.getenv("NEO4J_URI", REMOTE_MIDDLEWARE_HOST),
    "NEO4J_USER": os.getenv("NEO4J_USER"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
}

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "ELASTICSEARCH_URL_LIST": [os.getenv("ELASTICSEARCH_URL")],
    "API_KEY": os.getenv("ELASTICSEARCH_API_KEY", "No-Key-Required"),
}

# Java Micro Service URLs
# 上游 Java 微服务的根地址，缺省时回落到本地占位，避免 import 期 "None + str" 直接抛异常
def _service_base_url(env_name: str, default: str) -> str:
    return (os.getenv(env_name) or default).rstrip("/")


JAVA_MICRO_SERVICE_URL = {
    "speech-recognize-job-done-notify-url": _service_base_url(
        "MICRO_LLM_BACKEND_CORE_BASE_URL", "http://localhost:11600") + "/api/model-call-task/notify-speech-recognize-job-done",
    "fetch-and-parse-html-url": _service_base_url(
        "MICRO_LLM_BACKEND_WEBSEARCH_ENGINE_BASE_URL", "http://localhost:11800") + "/api/web-html-retriever/fetch-and-parse-urls",
    "search-results": _service_base_url(
        "MICRO_LLM_BACKEND_WEBSEARCH_ENGINE_BASE_URL", "http://localhost:11800") + "/api/web-html-retriever/search-results",
    "check-user-has-enough-money": _service_base_url(
        "MICRO_LLM_BACKEND_PAYMENT_ORDER_BASE_URL", "http://localhost:11900") + "/api/user-usage-info/has-enough-money-for-service",
    "count-user-token-usage": _service_base_url(
        "MICRO_LLM_BACKEND_PAYMENT_ORDER_BASE_URL", "http://localhost:11900") + "/api/model-payment-txn/request-model-payment-txn",
    "notice-version-checkout": _service_base_url(
        "MICRO_LLM_BACKEND_CORE_BASE_URL", "http://localhost:11600") + "/api/knowledge-version/notice-version-checkout",
    "notice-rag-check-result": _service_base_url(
        "MICRO_LLM_BACKEND_CORE_BASE_URL", "http://localhost:11600") + "/api/rag-checker/notice-evaluate-result",
    "request-client-access-token": _service_base_url(
        "MICRO_LLM_BACKEND_USER_BASE_URL", "http://localhost:11750") + "/api/oauth2/token",
    "get-build-task-files-detail": _service_base_url(
        "MICRO_LLM_BACKEND_CORE_BASE_URL", "http://localhost:11600") + "/api/knowledge-index/get-build-task-files-detail",
    "get-build-task-file-detail": _service_base_url(
        "MICRO_LLM_BACKEND_CORE_BASE_URL", "http://localhost:11600") + "/api/knowledge-index/get-build-task-file-detail"
}

# Agent Configuration
AGENT_MAX_EXECUTION_TIME = int(os.getenv("AGENT_MAX_EXECUTION_TIME", 300))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 20))

# VPN Proxy Configuration（不配置则不使用代理）
VPN_PROXY_HTTP_URL = os.getenv("VPN_PROXY_URL")

# Web Search Embedding Models Weights
WEB_SEARCH_EMBEDDING_MODELS_WEIGHTS = {
    "EMBEDDING-001": float(os.getenv("WEB_SEARCH_EMBEDDING_MODEL_WEIGHT_001", 0.2)),
    "EMBEDDING-002": float(os.getenv("WEB_SEARCH_EMBEDDING_MODEL_WEIGHT_002", 0.2)),
    "EMBEDDING-003": float(os.getenv("WEB_SEARCH_EMBEDDING_MODEL_WEIGHT_003", 0.6)),
}
WEB_SEARCH_COUNT = 20

# Embedding Models Configuration
EMBEDDING_MODELS = [
    {
        "id": os.getenv("EMBEDDING_MODEL_PRODUCT_ID", "EMBEDDING-001"),
        "product_type": 5,
        "model_name": os.getenv("DEFAULT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
        "batch_size": int(os.getenv("EMBEDDING_001_BATCH_SIZE", 1)),
        "sub_inputs_batch_size": 32,
        "chunk_size": 3276,  # 1个中文字符约等于 1.3 到 2.5 个 tokens
        "embedding_ctx_length": 8191,
        "api_base": os.getenv("EMBEDDING_001_API_BASE", os.getenv("EMBEDDING_API_BASE", "http://localhost:48077/v1")),
        "api_key": os.getenv("EMBEDDING_API_KEY", "No-Key-Required"),
        "max_retries": int(os.getenv("EMBEDDING_001_MAX_RETRIES", 2)),
        "dimensions": None,
        "default": True
    },
    {
        "id": "EMBEDDING-002",
        "product_type": 5,
        "model_name": "text-embedding-v4",
        "batch_size": int(os.getenv("EMBEDDING_002_BATCH_SIZE", 6)),
        "sub_inputs_batch_size": 10,
        "chunk_size": 3276,  # 1个中文字符约等于 1.3 到 2.5 个 tokens
        "embedding_ctx_length": 8191,
        "api_base": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
        "max_retries": int(os.getenv("EMBEDDING_002_MAX_RETRIES", 2)),
        "dimensions": 2048,
        "extra_header": {
            "Host": "dashscope.aliyuncs.com",
        },
        "model_kwargs": {
            "encoding_format": "float"
        },
        "default": False
    },
    {
        "id": "EMBEDDING-003",
        "product_type": 5,
        "model_name": "embedding-3",
        "batch_size": int(os.getenv("EMBEDDING_003_BATCH_SIZE", 3)),
        "sub_inputs_batch_size": 32,
        "chunk_size": 1228,  # 1个中文字符约等于 1.3 到 2.5 个 tokens
        "embedding_ctx_length": 3071,
        "api_base": os.getenv("GLM_4_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
        "max_retries": int(os.getenv("EMBEDDING_003_MAX_RETRIES", 2)),
        "dimensions": 2048,
        "default": False
    },
]

# NLTK Data Path
NLTK_DATA_PATH = os.getenv("NLTK_DATA_PATH", "/usr/local/share/nltk_data")

# llm model configs
SUPPORTED_MODELS = [
    {
        "id": "MODEL-0002",
        "name": "qwen3.6-plus",
        "alias": "qwen-plus",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
    },
    {
        "id": "MODEL-0004",
        "name": "qwen3-max-preview",
        "alias": "qwen-max",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 262144,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
        "default_build_knowledge_product_id": True
    },
    {
        "id": "MODEL-0005",
        "name": "qwen-long",
        "alias": "qwen-long",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 10000000,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
    },
    {
        "id": "MODEL-0007",
        "name": "qwen/qwen3-coder:free",
        "alias": "qwen3-coder",
        "request_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 262100,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0008",
        "name": "qwen-flash",
        "alias": "qwen-flash",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 10000000,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
    },
    {
        "id": "MODEL-0602:free",
        "name": "deepseek/deepseek-r1-0528:free",
        "alias": "deepseek-r1",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 64000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0605",
        "name": "deepseek-reasoner",
        "alias": "deepseek-v3.2-exp:reasoner",
        "openai_api_base": os.getenv("DEEPSEEK_CHAT_COMPLETIONS_URL", "https://api.deepseek.com"),
        "context_tokens_limit": 128000,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
    {
        "id": "MODEL-0606",
        "name": "deepseek-chat",
        "alias": "deepseek-v3.2-exp",
        "openai_api_base": os.getenv("DEEPSEEK_CHAT_COMPLETIONS_URL", "https://api.deepseek.com"),
        "context_tokens_limit": 128000,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "reasoner_model_id": "MODEL-0605"
    },
    {
        "id": "MODEL-0212",
        "name": "glm-5",
        "alias": "GLM-5",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 131072,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0211",
        "name": "glm-4.7",
        "alias": "GLM-4.7",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 200000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0207",
        "name": "glm-4.6",
        "alias": "GLM-4.6",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 200000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0208",
        "name": "glm-4.5-x",
        "alias": "GLM-4.5-X",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 128000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0209",
        "name": "glm-4.5-air",
        "alias": "GLM-4.5-Air",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 128000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0210",
        "name": "glm-4.5-flash",
        "alias": "GLM-4.5-Flash:free",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 128000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0411",
        "name": "openai/gpt-4o-mini",
        "alias": "gpt-4o-mini",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 128000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0412",
        "name": "openai/gpt-4o",
        "alias": "gpt-4o",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 128000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0413",
        "name": "openai/gpt-5",
        "alias": "gpt-5",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0414",
        "name": "openai/gpt-oss-20b:free",
        "alias": "gpt-oss-20b:free",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1311000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0415",
        "name": "openai/gpt-5.1",
        "alias": "gpt-5.1",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0416",
        "name": "openai/gpt-5-mini",
        "alias": "gpt-5-mini",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0417",
        "name": "openai/gpt-5-nano",
        "alias": "gpt-5-nano",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0418",
        "name": "openai/gpt-5.2",
        "alias": "gpt-5.2",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0515",
        "name": "anthropic/claude-sonnet-4.5",
        "alias": "claude-sonnet-4.5",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0516",
        "name": "anthropic/claude-haiku-4.5",
        "alias": "claude-haiku-4.5",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 200000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0517",
        "name": "anthropic/claude-opus-4.5",
        "alias": "claude-opus-4.5",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 200000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0518",
        "name": "anthropic/claude-opus-4.6",
        "alias": "claude-opus-4.5",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 200000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0519",
        "name": "anthropic/claude-sonnet-4.6",
        "alias": "claude-sonnet-4.6",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0303",
        "name": "google/gemini-2.0-flash-exp:free",
        "alias": "gemini-2.0-flash-experimental",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0305",
        "name": "google/gemini-2.5-flash",
        "alias": "gemini-2.5-flash",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0306",
        "name": "google/gemini-2.5-pro",
        "alias": "gemini-2.5-pro",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0307",
        "name": "google/gemini-2.5-flash-lite",
        "alias": "gemini-2.5-flash-lite",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0308",
        "name": "google/gemini-3-pro-preview",
        "alias": "gemini-3-pro-preview",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0309",
        "name": "google/gemini-3-flash-preview",
        "alias": "gemini-3-flash-preview",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0310",
        "name": "google/gemini-3.1-pro-preview",
        "alias": "gemini-3.1-pro-preview",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1048576,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0311",
        "name": "google/gemini-3.1-flash-lite-preview",
        "alias": "gemini-3.1-flash-lite-preview",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1050000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0419",
        "name": "openai/gpt-5.4",
        "alias": "gpt-5.4",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1050000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0420",
        "name": "openai/gpt-5.4-mini",
        "alias": "gpt-5.4-mini",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 400000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0701",
        "name": "x-ai/grok-4.1-fast",
        "alias": "grok-4.1-fast",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 200000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0702",
        "name": "x-ai/grok-4.20",
        "alias": "grok-4.20",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 2000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0421",
        "name": "openai/gpt-5.5",
        "alias": "gpt-5.5",
        "openai_api_base": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1050000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0520",
        "name": "anthropic/claude-opus-4.7",
        "alias": "claude-opus-4.7",
        "anthropic_api_url": os.getenv("OPEN_ROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1"),
        "provider": "openrouter",
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("OPEN_ROUTER_API_KEY"),
    },
    {
        "id": "MODEL-0607",
        "name": "deepseek-v4-flash",
        "alias": "deepseek-v4-flash",
        "openai_api_base": os.getenv("DEEPSEEK_CHAT_COMPLETIONS_URL", "https://api.deepseek.com"),
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
    {
        "id": "MODEL-0608",
        "name": "deepseek-v4-pro",
        "alias": "deepseek-v4-pro",
        "openai_api_base": os.getenv("DEEPSEEK_CHAT_COMPLETIONS_URL", "https://api.deepseek.com"),
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
    {
        "id": "MODEL-0213",
        "name": "glm-5.1",
        "alias": "GLM-5.1",
        "request_url": os.getenv("GLM_4_PLUS_CHAT_COMPLETIONS_URL",
                                 "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "context_tokens_limit": 200000,
        "api_key": os.getenv("GLM_4_PLUS_API_KEY"),
    },
    {
        "id": "MODEL-0010",
        "name": "qwen3.6-max-preview",
        "alias": "qwen3.6-max-preview",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 262144,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
    },
    {
        "id": "MODEL-0011",
        "name": "qwen3.6-plus",
        "alias": "qwen3.6-plus",
        "request_url": os.getenv("QWEN_PLUS_CHAT_COMPLETIONS_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "context_tokens_limit": 1000000,
        "api_key": os.getenv("QWEN_SERIES_API_KEY"),
    },
]

# embedding model configs
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[0]
EMBEDDING_MINI_BATCH_SIZE = 4
EMBEDDING_MINI_BATCH_SIZE_FOR_SMALLER_DOCS = 256

# common configs
KB_ROOT_PATH = VS_ROOT_PATH
MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR = KB_ROOT_PATH

# file loader configs, ~ 2k tokens per chunk
# cn_chunk_size: 1250 chars ≈ 2000 tokens
# en_chunk_size: 8000 chars ≈ 2000 tokens
CHARACTER_TEXT_SPLITTER = {
    "cn_chunk_size": 1250,
    "cn_chunk_overlap": 0,
    "en_chunk_size": 7500,
    "en_chunk_overlap": 0,
    "child_cn_chunk_size": 115,
    "child_cn_chunk_overlap": 10,
    "child_en_chunk_size": 625,
    "child_en_chunk_overlap": 30,
}

DEFAULT_SEPARATOR = {
    "cn": [r";", r"；", r".", r"!", r"?", r"。", r"！", r"\?", r"\.{6}", r"\…{2}", r"\…{3}", r"\n", r"\n\n", r" "],
    "en": [r";", r".", r"!", r"?", r"\n", r"\n\n", r" "],
}
DEFAULT_REPLACE_WORDS_DICT = {
    r"\n": "",
    r"\s": "",
}

# prompt template configs
DEFAULT_SERIALIZED_EN_PROMPT_TEMPLATE = "You are a helpful assistant."
DEFAULT_SERIALIZED_CN_PROMPT_TEMPLATE = "你是一个乐于助人的助手，请用中文回答用户的问题。"

# hippo rag configs
EXTRACT_OPENIE_FROM_DOCS_PROCESSOR_NUM = 10

# RAG checker configs
RAG_CHECKER_DIR_NAME = "rag_checker"
CREATE_INDEX_RETRYABLE_CONTEXT_FILE_DIR_NAME = "retryable_context_dir"
INDEX_ERROR_MESSAGE_DIR_NAME = "index_error_message_dir"

# BM25 index configs
BM25_INDEX_ROOT_PATH = KB_ROOT_PATH

BUILDING_BUNDLE_OF_INDEX_TASK_CONFIG = {
    "consume_task_count": 2,
    "consume_task_interval_seconds": 2,
    "consume_build_kg_index_queue_name": "BUILD_KNOWLEDGE_INDEX_TASK_QUEUE",
    "consume_standalone_build_kg_index_queue_name": "STAND_ALONE_KNOWLEDGE_INDEX_TASK_QUEUE",
}

SUMMARY_CHUNK_SIZE = 27500 * 2  # 不直接区分中英文，统一按一个中文字符约等于2个tokens来计算
SUMMARY_CHUNK_OVERLAP_TOKENS = 0
SUMMARY_MAP_ANALYSIS_PRODUCT_ID = "MODEL-0008"
SUMMARY_MAP_ANALYSIS_MODEL_CONFIG = {
    "temperature": 0.1,
    "streaming": False,
    "max_tokens": 1800,  # 300 core summary * (4 + 2) approximately
    "reasoning_exclude": True,
}
SUMMARY_REDUCE_ANALYSIS_PRODUCT_ID = "MODEL-0004"
SUMMARY_REDUCE_ANALYSIS_MODEL_CONFIG = {
    "temperature": 0.1,
    "streaming": False,
    "max_tokens": 4800,  # 800 core summary * (4 + 2) approximately
    "reasoning_exclude": True,
}

DIRECT_SUMMARY_TOKEN_THRESHOLD = SUMMARY_CHUNK_SIZE
DOCUMENT_ANALYSIS_MODEL_PRODUCT_ID = "MODEL-0008"
DOCUMENT_ANALYSIS_MODEL_CONFIG = {
    "temperature": 0.1,
    "streaming": False,
    "max_tokens": 2400,
    "reasoning_exclude": True,
}
MINIMAL_DOCUMENT_ANALYSIS_CHUNK_SIZE_RATIO = 1  # 进行文档分析的最小文本token数量应该至少是chunk_size的x倍

AAD_INFO = {
    "grant_type": "client_credentials",
    "client_id": os.getenv("AAD_CLIENT_ID", "lingro-rag-service"),
    "client_secret": os.getenv("LINGRO_LANGCHAIN_PY_CLIENT_SECRET"),
    "scopes": "mcp:read,mcp:write"
}

MCP_REQUEST_CONFIG = {
    "request_timeout": 90,  # seconds
}

CHAT_SESSION_TEMP_FILE_DIR_NAME = "chat_session_uploaded_files"
CHAT_SESSION_TEMP_FILE_BUILD_KNOWLEDGE_INDEX_CONFIG = {
    "mode": "elements",
    "chunking_strategy": "by_title",
    "distance_strategy": "EUCLIDEAN_DISTANCE",
    "build_kg_index_product_id": "MODEL-0004"
}

# Server Timezone Configuration
SERVER_TIMEZONE = os.getenv("SERVER_TIMEZONE", "Asia/Shanghai")
