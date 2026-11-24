from typing import Optional
from pydantic import BaseModel, Field


class ModelPaymentTxnRequestVO(BaseModel):
    """
    模型支付交易请求VO
    对应Java端的ModelPaymentTxnRequestVO
    """
    user_id: str = Field(default=None, description="用户ID", alias="userId")
    product_id: str = Field(default=None, description="产品ID", alias="productId")
    product_type: int = Field(default=None, description="产品类型: 1-LLM, 5-Embedding", alias="productType")
    usage: Optional[int] = Field(default=None, description="语音识别消耗的时长(秒)")
    third_party_id: Optional[str] = Field(default=None, description="第三方请求ID", alias="thirdPartyId")
    prompt_tokens: Optional[int] = Field(default=None, description="输入token数", alias="promptTokens")
    completion_tokens: Optional[int] = Field(default=None, description="输出token数", alias="completionTokens")
    cache_creation_tokens: Optional[int] = Field(default=None, description="缓存写入的token数", alias="cacheCreationTokens")
    cache_read_tokens: Optional[int] = Field(default=None, description="缓存读取的token数", alias="cacheReadTokens")
    total_tokens: int = Field(..., description="总token数", alias="totalTokens")
    usage_metadata: Optional[str] = Field(default=None, description="使用元数据JSON字符串", alias="usageMetadata")

