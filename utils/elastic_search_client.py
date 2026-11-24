from typing import List

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from configs.model_config import ELASTICSEARCH_CONFIG, logger, EMBEDDING_MINI_BATCH_SIZE
from utils.common_util import get_beijing_current_time


class ElasticSearchClient:
    def __init__(self,
                 embedding_model: Embeddings = None,
                 hosts=ELASTICSEARCH_CONFIG["ELASTICSEARCH_URL_LIST"],
                 api_key=ELASTICSEARCH_CONFIG["API_KEY"]):
        assert embedding_model is not None, "embeddings not initialized"
        self.hosts = hosts
        self.client = Elasticsearch(hosts=hosts, verify_certs=False, api_key=api_key)
        self.embedding_model = embedding_model

    @property
    def es_client(self):
        return self.client

    def find_indices_with_prefix(self, index_name_prefix: str):
        try:
            # 获取所有索引
            indices = self.client.cat.indices(format='json')
            # 过滤出以index_name_prefix开头的索引
            matching_indices = [index['index'] for index in indices if index['index'].startswith(index_name_prefix)]
            return matching_indices
        except Exception as e:
            print(f"Error finding indices with prefix {index_name_prefix}: {e}")
            return []

    def create_index(self, index_name: str, mappings: dict = None, is_chinese_text=False):
        if self.client.indices.exists(index=index_name):
            logger.info(
                f"[create_index]-Index {index_name} already exists and will not be created again in elastic search.")
            return
        if not mappings:
            if is_chinese_text:
                """
                For Chinese text, we use the ik_max_word tokenizer for the page_content field, and the english analyzer for the english sub-field.
                When searching, we use the ik_smart tokenizer for the page_content field, and the english analyzer for the english sub-field.
                """
                page_context_settings = {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                    "fields": {
                        "english": {
                            "type": "text",
                            "analyzer": "english"  # 英文子字段分词
                        }
                    }
                }
            else:
                page_context_settings = {
                    "type": "text",
                }
            mappings = {
                "dynamic": True,
                "properties": {
                    "prefix_id": {
                        "type": "keyword",
                    },
                    "doc_id": {
                        "type": "keyword",
                    },
                    "page_content": page_context_settings,
                    "dense_vector_field": {
                        "type": "dense_vector",
                        "index": True,
                        "index_options": {
                            "type": "int8_hnsw"
                        }
                    },
                    "metadata": {
                        "type": "object",
                    },
                    "created_at": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss",
                    },
                }
            }
        return self.client.indices.create(index=index_name, mappings=mappings)

    def add_or_update_index_data(self, index_name: str,
                                 docs: list[Document],
                                 refresh: bool = True,
                                 **kwargs):
        mini_batch_size = kwargs.get("mini_batch_size", EMBEDDING_MINI_BATCH_SIZE)
        request_bodies = []
        mini_batch_docs = []
        for i, doc in enumerate(docs):
            if doc is None:
                continue
            if doc.page_content is None:
                continue

            mini_batch_docs.append(doc)
            if len(mini_batch_docs) == mini_batch_size or i == len(docs) - 1:
                import time

                start_time = time.time()
                page_content_embeddings = self.embedding_model.embed_documents(
                    [doc.page_content for doc in mini_batch_docs])
                end_time = time.time()
                logger.info(f"Embedding documents took {end_time - start_time} seconds")

                for doc, embedding in zip(mini_batch_docs, page_content_embeddings):
                    metadata = {"index": {
                        "_index": index_name,
                        "_id": doc.id,
                    }}
                    data_body = {
                        "prefix_id": doc.metadata.get("prefix_id", None),
                        "doc_id": doc.id,
                        "page_content": doc.page_content,
                        "dense_vector_field": embedding,
                        "metadata": doc.metadata,
                        "created_at": doc.metadata.get("created_at",
                                                       get_beijing_current_time().strftime("%Y-%m-%d %H:%M:%S")),
                    }
                    request_bodies.append(metadata)
                    request_bodies.append(data_body)
                es_resp = self.client.bulk(operations=request_bodies)
                if es_resp.get("errors", False):
                    logger.error(f"[add_or_update_index_data]-Error in bulk indexing: {es_resp}")
                    raise Exception(f"Error in bulk indexing: {es_resp}")
                mini_batch_docs = []
                request_bodies = []

        if refresh:
            self.client.indices.refresh(index=index_name)

        return len(request_bodies)

    def delete_index(self, index_name: str):
        return self.client.indices.delete(index=index_name)

    def delete_documents_in_index(self, index_name: str, doc_ids: List[str], refresh: bool = True):
        """
        Delete a document from the specified index by its doc_id.
        """
        try:
            for doc_id in doc_ids:
                self.client.delete(index=index_name, id=doc_id)
                logger.info(f"[delete_document]-Document with id {doc_id} deleted from index {index_name}.")
            if refresh:
                self.client.indices.refresh(index=index_name)
        except Exception as e:
            logger.error(f"[delete_document]-Error deleting document from index {index_name}: {e}")
            raise e
