from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddingsWrapper(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        pass

    def get_sentence_transformers_client(self) -> SentenceTransformer:
        return self._client
