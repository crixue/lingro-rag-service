from typing import Union, List

import numpy as np
from langchain_core.embeddings import Embeddings


def normalize_embeddings(embeddings: Union[np.ndarray, list[list[float]]]) -> np.ndarray:
    if isinstance(embeddings, np.ndarray):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    else:
        embeddings = np.array(embeddings)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

class OpenAICompatibleEmbeddingWrapper:
    def __init__(self, embedding_model: Embeddings):
        assert embedding_model is not None, "embeddings not initialized"
        self.embedding = embedding_model

    def encode_text(self, text: Union[str, List], norm=False, **kwargs) -> np.ndarray:
        """
        请确认embedding模型本身是否已经做了norm处理，如果没有做norm处理，可以在这里做norm处理
        """
        queries = []
        if isinstance(text, str):
            queries.append(text)
        elif isinstance(text, list):
            queries.extend(text)
        for i, item in enumerate(queries):
            if item is None or len(item.strip()) == 0:
                queries[i] = "NOT_A_VALID_TEXT"

        embeddings= self.embedding.embed_documents(queries)
        embeddings = np.array(embeddings)
        if norm:
            embeddings = normalize_embeddings(embeddings)
        return embeddings
