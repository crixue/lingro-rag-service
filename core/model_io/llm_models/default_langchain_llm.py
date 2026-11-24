import logging
from typing import Optional, Dict, Any, Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel, Field

from configs.model_config import LOG_FORMAT, SUPPORTED_MODELS, DEFAULT_CONTEXT_TOKENS_LIMIT, VPN_PROXY_HTTP_URL
from core.callbacks.llm_callbacks import BaseLLMTokenUsageCallbackFunction, ZhipuTokenUsageCallbackFunction, \
    OpenRouterGeminiTokenUsageCallbackFunction
from core.model_io.prompts.react_prompts import BaseReactPrompt
from core.model_io.support_models_info_util import SupportedModelsInfoUtil

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


class DefaultLangchainLLM(BaseModel):
    llm_model_name: str = Field(alias="model_name")
    llm_model_config: Optional[Mapping[str, Any]] = Field(alias="model_config", default=None)
    llm_context_tokens_limit: Optional[int] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    product_id: Optional[str] = Field(default=None)
    llm_callback_client: Optional[BaseLLMTokenUsageCallbackFunction] = Field(default=None)

    def convert_model_config_in_request(self) -> Dict[str, Any]:
        sub_classes = BaseLLMConfigConverter.__subclasses__()
        for sub_class in sub_classes:
            if sub_class.if_supported_llm_model_name(self.llm_model_name):
                return sub_class(llm_model_name=self.llm_model_name,
                                 model_config=self.llm_model_config).convert_model_config()
        return BaseLLMConfigConverter(llm_model_name=self.llm_model_name,
                                      model_config=self.llm_model_config).convert_model_config()

    def backup_model_in_request(self) -> Dict[str, Any]:
        sub_classes = BaseLLMConfigConverter.__subclasses__()
        for sub_class in sub_classes:
            if sub_class.if_supported_llm_model_name(self.llm_model_name):
                return sub_class(llm_model_name=self.llm_model_name,
                                 model_config=self.llm_model_config).backup_model()
        return BaseLLMConfigConverter(llm_model_name=self.llm_model_name,
                                      model_config=self.llm_model_config).backup_model()

    @classmethod
    def llm_client_factory(cls,
                           llm_model_name: str,
                           llm_model_config: Optional[Mapping[str, Any]] = None,
                           user_id: Optional[str] = None,
                           product_id: Optional[str] = None,
                           llm_callback_client: Optional[BaseLLMTokenUsageCallbackFunction] = None,
                           additional_llm_config: Optional[Mapping[str, Any]] = None,
                           **kwargs) -> BaseChatModel | None:
        llm = cls(model_name=llm_model_name,
                  model_config=llm_model_config,
                  user_id=user_id,
                  product_id=product_id,
                  llm_callback_client=llm_callback_client,
                  )
        return llm.get_llm_client(additional_llm_config=additional_llm_config)

    def get_llm_client(self,
                       additional_llm_config: Optional[Mapping[str, Any]] = None,
                       **kwargs) -> BaseChatModel | None:
        llm = None
        self.llm_callback_client = BaseLLMTokenUsageCallbackFunction(user_id=self.user_id, product_id=self.product_id)

        model_kwargs = self.convert_model_config_in_request()
        model_dict = SupportedModelsInfoUtil.find_supported_model_by_id(self.product_id)
        backup_model_dict = self.backup_model_in_request()

        if additional_llm_config is not None:
            model_kwargs.update(additional_llm_config)

        if self.llm_model_name.count("qwen") > 0:
            from core.model_io.llm_models.tongyi_qwen_v4 import ChatTongyi
            llm = ChatTongyi(model_name=self.llm_model_name,
                             api_key=model_dict.get("api_key"),
                             api_base=model_dict.get("request_url"),
                             callbacks=[self.llm_callback_client],
                             **model_kwargs,
                             )
        elif self.llm_model_name.count("glm-") > 0:
            self.llm_callback_client = ZhipuTokenUsageCallbackFunction(user_id=self.user_id, product_id=self.product_id)
            from core.model_io.llm_models.zhipu_glm_v3 import ChatZhipuAI
            llm = ChatZhipuAI(model_name=self.llm_model_name,
                              zhipuai_api_base=model_dict.get("request_url"),
                              zhipuai_api_key=model_dict.get("api_key"),
                              callbacks=[self.llm_callback_client],
                              **model_kwargs,
                              )
        elif self.llm_model_name.count("gemini-") > 0:
            provider = model_dict.get("provider", None)
            if provider is None or provider.lower() == "google":
                from core.model_io.llm_models.google_gemini_v4 import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=self.llm_model_name,
                                             transport='rest',
                                             request_url=model_dict.get("request_url"),
                                             google_api_key=model_dict.get("api_key"),
                                             callbacks=[self.llm_callback_client],
                                             **model_kwargs
                                             )
            elif provider is not None and provider.lower() == "openrouter":
                self.llm_callback_client = OpenRouterGeminiTokenUsageCallbackFunction(user_id=self.user_id,
                                                                                      product_id=self.product_id)
                from core.model_io.llm_models.openrouter_google_gemini_v4 import ChatGoogleGemini
                llm = ChatGoogleGemini(
                    model_name=model_dict.get("name"),
                    openai_api_base=model_dict.get("openai_api_base"),
                    openai_api_key=model_dict.get("api_key"),
                    openai_proxy=VPN_PROXY_HTTP_URL,
                    callbacks=[self.llm_callback_client],
                    # model_kwargs=openrouter_reasoning_model_config(self.llm_model_config),
                    **model_kwargs,
                )
        elif self.llm_model_name.count("gpt-") > 0:
            # from core.model_io.llm_models.openrouter_openai_gpt_v4 import ChatOpenAI
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model_name=model_dict.get("name"),
                openai_api_base=model_dict.get("openai_api_base"),
                openai_api_key=model_dict.get("api_key"),
                openai_proxy=VPN_PROXY_HTTP_URL,
                callbacks=[self.llm_callback_client],
                **model_kwargs,
            )
        elif self.llm_model_name.count("claude-") > 0:
            from core.model_io.llm_models.openrouter_claude_v4 import ChatClaudeAI
            llm = ChatClaudeAI(
                model_name=model_dict.get("name"),
                openai_api_base=model_dict.get("anthropic_api_url"),
                openai_api_key=model_dict.get("api_key"),
                openai_proxy=VPN_PROXY_HTTP_URL,
                callbacks=[self.llm_callback_client],
                **model_kwargs,
            )
        elif self.llm_model_name.count("deepseek-") > 0:
            # from core.model_io.llm_models.deepseek_v4 import ChatDeepSeek
            from langchain_deepseek import ChatDeepSeek
            model_name = model_dict.get("name")
            back_up_model_name = backup_model_dict.get("name", None)
            if back_up_model_name is not None:
                model_name = back_up_model_name
            llm = ChatDeepSeek(
                model_name=model_name,
                api_base=model_dict.get("openai_api_base"),
                api_key=model_dict.get("api_key"),
                callbacks=[self.llm_callback_client],
                **model_kwargs,
            )
        elif self.llm_model_name.count("grok") > 0:
            from core.model_io.llm_models.openrouter_xai_grok_v4 import ChatGrokAI
            llm = ChatGrokAI(
                model_name=model_dict.get("name"),
                openai_api_base=model_dict.get("openai_api_base"),
                openai_api_key=model_dict.get("api_key"),
                openai_proxy=VPN_PROXY_HTTP_URL,
                callbacks=[self.llm_callback_client],
                **model_kwargs,
            )
        logger.debug(
            f"[DefaultLangchainLLM]-Loading {self.llm_model_name} model start with config {model_kwargs}...")
        if llm is None:
            logger.error(f"[DefaultLangchainLLM]-Unsupported llm type {self.llm_model_name}")
            raise ValueError(f"Unsupported llm type {self.llm_model_name}")
        return llm

    def get_react_system_prompt(self,
                                have_context: bool = True,
                                is_chinese_text: bool = True,
                                have_plugins: bool = False,
                                ) -> BasePromptTemplate or None:
        sub_react_prompt_classes = BaseReactPrompt.__subclasses__()
        for sub_cls in sub_react_prompt_classes:
            if self.llm_model_name.startswith(sub_cls().model_start_name):
                return sub_cls.get_react_system_prompt(have_context=have_context,
                                                       is_chinese_text=is_chinese_text,
                                                       have_plugins=have_plugins)
        return BaseReactPrompt.get_react_system_prompt(have_context=have_context,
                                                       is_chinese_text=is_chinese_text,
                                                       have_plugins=have_plugins)

    @property
    def context_tokens_limit(self):
        self.llm_context_tokens_limit = DEFAULT_CONTEXT_TOKENS_LIMIT
        for model_args in SUPPORTED_MODELS:
            if model_args.get("name", None) == self.llm_model_name:
                self.llm_context_tokens_limit = model_args.get("context_tokens_limit", DEFAULT_CONTEXT_TOKENS_LIMIT)
                break
        return self.llm_context_tokens_limit


class BaseLLMConfigConverter(BaseModel):
    llm_model_name: str
    llm_model_config: Optional[dict] = Field(alias="model_config", default=None)

    @property
    def llm_model_name(self) -> str:
        return self.llm_model_name

    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return False

    @classmethod
    def alias_model_name(cls, llm_model_name) -> str:
        return SupportedModelsInfoUtil.find_supported_model_by_name(llm_model_name).get("alias", "").lower()

    def reasoning_enabled(self) -> bool:
        if self.llm_model_config is None:
            return True
        return self.llm_model_config.get("reasoning_enabled", True)

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}

        model_kwargs = {"streaming": self.llm_model_config.get("streaming", True),
                        "max_tokens": self.llm_model_config.get("max_tokens", 512),
                        "temperature": self.llm_model_config.get("temperature", 0.8),
                        # "repetition_penalty": self.llm_model_config.get("repetition_penalty", 1.1),
                        **(openrouter_reasoning_model_config(self.llm_model_config)),
                        }
        return model_kwargs

    def backup_model(self) -> Dict[str, Any]:
        return {}


def openrouter_reasoning_model_config(model_config: Optional[dict]) -> Dict[str, Any]:
    """
    暂时openAI chat模型不支持这些额外的推理参数
    """
    reasoning_enabled = model_config.get("reasoning_enabled", True)
    if reasoning_enabled:
        reasoning_model_config = {
            "reasoning": {
                "effort": model_config.get("reasoning_effort", "medium"),
                "max_tokens": model_config.get("reasoning_max_tokens", 800) if model_config.get("reasoning_effort",
                                                                                                "medium") is None else None,
                "exclude": model_config.get("reasoning_exclude", False),
            }
        }
    else:
        reasoning_model_config = {
            "reasoning": {
                "enabled": False,
            }
        }
    return reasoning_model_config


class ChatGLMModelVer1ConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return cls.alias_model_name(llm_model_name).count("glm-") > 0

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}

        model_kwargs = {"streaming": self.llm_model_config.get("streaming", True),
                        "max_tokens": self.llm_model_config.get("max_tokens", 512),
                        "temperature": self.llm_model_config.get("temperature", 0.8),
                        # "repetition_penalty": self.llm_model_config.get("repetition_penalty", 1.1),
                        "enable_thinking": self.reasoning_enabled(),
                        }
        return model_kwargs


class ChatClaudeConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return cls.alias_model_name(llm_model_name).count("claude-") > 0

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}

        reasoning_params = {}
        if self.reasoning_enabled():
            reasoning_params = {
                "reasoning": {
                    "effort": self.llm_model_config.get("reasoning_effort", "medium"),
                    "max_tokens": self.llm_model_config.get("reasoning_max_tokens", 800) if self.llm_model_config.get(
                        "reasoning_effort",
                        "medium") is None else None,
                    "exclude": self.llm_model_config.get("reasoning_exclude", False),
                }
            }
        model_kwargs = {
            "streaming": self.llm_model_config.get("streaming", True),
            "max_tokens": self.llm_model_config.get("max_tokens", 512),
            "temperature": self.llm_model_config.get("temperature", 0.8),
            "stream_usage": True,
            "extra_body": {
                **reasoning_params,
            }
        }
        return model_kwargs


class QwenConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return cls.alias_model_name(llm_model_name).count("qwen") > 0

    def convert_model_config(self) -> Dict[str, Any]:
        enable_thinking = self.reasoning_enabled()
        thinking_budget = self.llm_model_config.get("reasoning_max_tokens", 800) if enable_thinking else None
        model_kwargs = {"streaming": self.llm_model_config.get("streaming", True),
                        "max_tokens": self.llm_model_config.get("max_tokens", 5000),
                        "enable_thinking": enable_thinking,
                        "thinking_budget": thinking_budget,
                        # "top_p": self.llm_model_config.get("top_p", 0.7)
                        }
        temperature = self.llm_model_config.get("temperature", 0.85)  # qwen系列 取值范围：[0, 2)，系统默认值0.85。不建议取值为0，无意义。
        model_kwargs["temperature"] = temperature * 1.7
        # model_kwargs["repetition_penalty"] = self.llm_model_config.get("repetition_penalty", 1.1)
        return model_kwargs


class ChatGoogleGenAIConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        model = SupportedModelsInfoUtil.find_supported_model_by_name(llm_model_name)
        provider = model.get("provider", None)
        return cls.alias_model_name(llm_model_name).count("gemini") > 0 and provider != "openrouter"

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}
        model_kwargs = {"candidate_count": 1,
                        "max_output_tokens": self.llm_model_config.get("max_tokens", 5000),
                        "streaming": self.llm_model_config.get("streaming", True),
                        # "top_p": self.llm_model_config.get("top_p", 0.95),
                        "temperature": self.llm_model_config.get("temperature", 0.8)
                        }
        return model_kwargs


class Gpt4OConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return cls.alias_model_name(llm_model_name).count("gpt-") > 0

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}
        model_kwargs = {
            "max_tokens": self.llm_model_config.get("max_tokens", 5000),
            "streaming": self.llm_model_config.get("streaming", True),
            # "top_p": self.model_config.get("top_p", 0.90),
            "temperature": self.llm_model_config.get("temperature", 0.8),
            **openrouter_reasoning_model_config(self.llm_model_config)
        }
        return model_kwargs


class DeepSeekConfigConverter(BaseLLMConfigConverter):
    @classmethod
    def if_supported_llm_model_name(cls, llm_model_name) -> bool:
        return cls.alias_model_name(llm_model_name).count("deepseek-") > 0

    def convert_model_config(self) -> Dict[str, Any]:
        if self.llm_model_config is None:
            return {}
        model_kwargs = {
            "max_tokens": min(self.llm_model_config.get("max_tokens", 5000), 8000),  # DeepSeek单次调用最大支持8000
            "streaming": self.llm_model_config.get("streaming", True),
            # "top_p": self.model_config.get("top_p", 0.90),
            "temperature": self.llm_model_config.get("temperature", 0.8)
        }
        return model_kwargs

    def backup_model(self) -> Dict[str, Any]:
        """
        Deepseek 推理功能使用的是 alias 带:reasoning的模型
        """
        apply_tools = self.llm_model_config.get("apply_tools", True)  # 目前V3.2模型:reasoning推理模型不支持使用工具
        if self.reasoning_enabled() and not apply_tools:
            current_model_info = SupportedModelsInfoUtil.find_supported_model_by_name(self.llm_model_name)
            reasoner_model_id = current_model_info.get("reasoner_model_id", None)
            if reasoner_model_id is not None:
                backup_model = SupportedModelsInfoUtil.find_supported_model_by_id(model_id=reasoner_model_id)
                return backup_model
        return {}
