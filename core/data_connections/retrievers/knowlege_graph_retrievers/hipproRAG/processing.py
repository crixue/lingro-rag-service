import json
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_file_name(path):
    return path.split('/')[-1].replace('.jsonl', '').replace('.json', '')


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def mean_pooling_embedding(input_str: str, tokenizer, model, device='cuda'):
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)

    embedding = mean_pooling(outputs[0], inputs['attention_mask']).to('cpu').detach().numpy()
    return embedding


def mean_pooling_embedding_with_normalization(input_str, tokenizer, model, device='cuda'):
    encoding = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs[0], attention_mask)
    embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T

    return embeddings

def mean_pooling_v2(model_output, attention_mask):
    token_embeddings = model_output.data['token_embeddings']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def mean_pooling_embedding_v2(input_str: str | list[str], sentence_transformer_model: SentenceTransformer, device='cuda'):
    intput_str = [input_str] if isinstance(input_str, str) else input_str
    with torch.no_grad():
        model = sentence_transformer_model.to(device)
        encoded_input = model.tokenizer(intput_str, return_tensors='pt', padding=True, truncation=True).to(device)
        model_output = model(encoded_input)
        return mean_pooling_v2(model_output, encoded_input['attention_mask'])


def mean_pooling_embedding_with_normalization_v2(input_str: str | list[str], sentence_transformer_model: SentenceTransformer, device='cuda'):
    input_str = [input_str] if isinstance(input_str, str) else input_str
    with torch.no_grad():
        model = sentence_transformer_model.to(device)
        encoded_input = model.tokenizer(input_str, return_tensors='pt', padding=True, truncation=True).to(device)
        model_output = model(encoded_input)
        embeddings = mean_pooling_v2(model_output, encoded_input['attention_mask'])
        embedding_with_normalization = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T
        return embedding_with_normalization


def processing_phrases(phrase):
    return re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', ' ', str(phrase).lower()).strip()


def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ''
    else:
        return ''


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if min_val == max_val:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)
