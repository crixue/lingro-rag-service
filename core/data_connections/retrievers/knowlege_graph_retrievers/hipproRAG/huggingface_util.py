from typing import Union, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch

from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import \
    mean_pooling_embedding_with_normalization, mean_pooling_embedding, mean_pooling_embedding_with_normalization_v2, \
    mean_pooling_embedding_v2


class HippoRAGEmbeddingWrapper:
    def __init__(self, model_name: str | None = None, sentence_transformer_model: SentenceTransformer | None = None, device='cuda'):
        self.device = device
        if sentence_transformer_model is not None:
            self.sentence_transformer_model = sentence_transformer_model
        else:
            self.model_name = model_name
            self.model_name_processed = model_name.replace('/', '_').replace('.', '_')
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_text(self, text: Union[str, List], instruction=None, norm=True, return_cpu=False, return_numpy=False):
        encoding_func = mean_pooling_embedding_with_normalization if norm else mean_pooling_embedding
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            res = []
            if len(text) > 1:
                for t in tqdm(text, total=len(text), desc=f"HF model {self.model_name} encoding"):
                    res.append(encoding_func(t, self.tokenizer, self.model, self.device))
            else:
                res = [encoding_func(text[0], self.tokenizer, self.model, self.device)]
            res = torch.stack(res)
            res = torch.squeeze(res, dim=1)

        if return_cpu:
            res = res.cpu()
        if return_numpy:
            res = res.numpy()
        return res


    def encode_text_v2(self, text: Union[str, List], instruction=None, norm=True, return_cpu=False, return_numpy=False):
        encoding_func = mean_pooling_embedding_with_normalization_v2 if norm else mean_pooling_embedding_v2

        if isinstance(text, str):
            text = [text]
        res = []
        if len(text) > 1:
            for t in tqdm(text, total=len(text)):
                res.append(encoding_func(t, self.sentence_transformer_model, self.device))
        else:
            res.append(encoding_func(text[0], self.sentence_transformer_model, self.device))

        res = torch.stack(res)
        res = torch.squeeze(res, dim=1)

        if return_cpu:
            res = res.cpu()
        if return_numpy:
            res = res.numpy()
        return res

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """

        @param query_vec: query vector
        @param doc_vecs: doc matrix
        @return: a matrix of query-doc scores
        """
        return np.dot(doc_vecs, query_vec.T)



if __name__ == '__main__':
    import torch

    # 创建一个包含多个张量的列表
    res = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]]), torch.tensor([[7, 8, 9]])]

    # 将列表中的张量堆叠成一个单一的张量
    res = torch.stack(res)
    print(res.shape)  # 输出: torch.Size([3, 1, 3])

    # 去除第1维度上的单维度条目
    res = torch.squeeze(res, dim=1)
    print(res.shape)  # 输出: torch.Size([3, 3])