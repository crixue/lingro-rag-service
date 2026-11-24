import unittest

from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2
from utils.elastic_search_client import ElasticSearchClient


class MyTestCase(unittest.TestCase):
    def test_es_client(self):
        elastic_search_client = ElasticSearchClient(embedding_model=DefaultEmbeddingsV2.get_static_embeddings(True)[0])
        elastic_search_client.create_index("test_index")

    def test_norm_embedding(self):
        import torch
        # 假设 embeddings 是一个形状为 [batch_size, embedding_dim] 的张量
        embeddings = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        # 计算L2范数
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        # 归一化
        normalized_embeddings = embeddings / norms
        print(normalized_embeddings)

