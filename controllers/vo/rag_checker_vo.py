from typing import Union, Optional

from pydantic import BaseModel, Field, model_validator

from controllers.vo.startup_llm_model_request_vo import UsedModelConfig, RetrieverConfig


class RagCheckerConfig(BaseModel):
    rating_product_id: str = Field(default=None)
    test_count: Optional[int] = Field(default=None)
    test_dataset_file_path: str = Field(default=None)
    saved_result_dir: str = Field(default=None)
    agent_max_iterations: Optional[int] = Field(default=3)
    max_concurrency: Optional[int] = Field(default=8)
    few_shot_samples: Optional[Union[list[str], str]] = Field(default=None)
    only_answer_questions: Optional[bool] = Field(default=False)
    only_evaluate_rag_results: Optional[bool] = Field(default=False)


class RAGEvaluateRequestVO(BaseModel):
    user_id: str = Field(default=None)
    task_id: str = Field(default=None)
    is_chinese_text: bool = Field(default=True)
    used_model_config: UsedModelConfig = Field(default=None)
    retriever_config: RetrieverConfig = Field(default=None)
    rag_checker_config: RagCheckerConfig = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def validate_before(cls, values):
        if values.get("retriever_config", None) is None:
            raise ValueError("retriever_config should not be None")
        if values.get("used_model_config", None) is None:
            raise ValueError("used_model_config should not be None")
        return values

    @model_validator(mode="after")
    def validate_after(self):
        self.retriever_config.is_chinese_text = self.is_chinese_text
        self.used_model_config.is_chinese_text = self.is_chinese_text
        return self