import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

from elasticsearch import Elasticsearch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_elasticsearch._utilities import with_user_agent_header
from langchain_elasticsearch.client import create_elasticsearch_client

from core.data_connections.retrievers.utils.retrieve_util import parse_query, PARSE_INFO_KEY_PHRASES, \
    PARSE_INFO_KEY_LIMITATION, PARSE_INFO_KEY_SYNTAX

logger = logging.getLogger(__name__)


def document_field_mapper(hit: Mapping[str, Any]) -> Document:
    metadata = hit["_source"]["metadata"]
    content = hit["_source"]["page_content"]
    doc_id = hit["_source"]["doc_id"]
    return Document(page_content=content, metadata=metadata, id=doc_id)


class DefaultElasticsearchRetriever(BaseRetriever):
    """
    Elasticsearch retriever

    Args:
        es_client: Elasticsearch client connection. Alternatively you can use the
            `from_es_params` method with parameters to initialize the client.
        index_name: The name of the index to query. Can also be a list of names.
        body_func: Function to create an Elasticsearch DSL query body from a search
            string. The returned query body must fit what you would normally send in a
            POST request the the _search endpoint. If applicable, it also includes
            parameters the `size` parameter etc.
        content_field: The document field name that contains the page content. If
            multiple indices are queried, specify a dict {index_name: field_name} here.
        document_mapper: Function to map Elasticsearch hits to LangChain Documents.

    For synchronous applications, use the ``ElasticsearchRetriever`` class.
    For asyhchronous applications, use the ``AsyncElasticsearchRetriever`` class.
    """

    es_client: Elasticsearch
    index_name: Union[str, Sequence[str]]
    body_func: Optional[Callable[[str], Dict]] = None
    content_field: Optional[Union[str, Mapping[str, str]]] = None
    document_mapper: Optional[Callable[[Mapping], Document]] = None
    embedding_model: Optional[Embeddings] = None
    query_kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.content_field is None and self.document_mapper is None:
            raise ValueError("One of content_field or document_mapper must be defined.")
        if self.content_field is not None and self.document_mapper is not None:
            raise ValueError(
                "Both content_field and document_mapper are defined. "
                "Please provide only one."
            )

        if not self.document_mapper:
            if isinstance(self.content_field, str):
                self.document_mapper = self._single_field_mapper
            elif isinstance(self.content_field, Mapping):
                self.document_mapper = self._multi_field_mapper
            else:
                raise ValueError(
                    "unknown type for content_field, expected string or dict."
                )

        self.es_client = with_user_agent_header(self.es_client, "langchain-py-r")

    @classmethod
    def from_es_params(
            cls,
            index_name: Union[str, Sequence[str]],
            body_func: Optional[Callable[[str], Dict]],
            content_field: Optional[Union[str, Mapping[str, str]]] = None,
            document_mapper: Optional[Callable[[Mapping], Document]] = None,
            url: Optional[str] = None,
            cloud_id: Optional[str] = None,
            api_key: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            params: Optional[Dict[str, Any]] = None,
            embedding_model: Optional[Embeddings] = None,
            query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DefaultElasticsearchRetriever":
        client = None
        try:
            client = create_elasticsearch_client(
                url=url,
                cloud_id=cloud_id,
                api_key=api_key,
                username=username,
                password=password,
                params=params,
            )
        except Exception as err:
            logger.error(f"Error connecting to Elasticsearch: {err}")
            raise err

        return cls(
            es_client=client,
            index_name=index_name,
            body_func=body_func,
            content_field=content_field,
            document_mapper=document_mapper,
            embedding_model=embedding_model,
            query_kwargs=query_kwargs,
        )

    def hybrid_query(self, query: str, **kwargs):
        """
        Parsed query is a list of phrases, limitations, and syntax items. Please refer prompt_definitions/builtin/lang/en/tool_definition_prompt.json
        phrases: `Core Entities: Use precise phrases or extended phrases for fuzzy search, with each phrase wrapped in the <SEP> tag.`
        limitations: `Contextual Limits: Field name:value (e.g., year:2023) and use <LIMIT> wrap each phrase.`
        syntax: `Logical Controls and use <SYN> tag to wrap each phrase. Exclude words by adding a - before them.`
        """
        top_k = kwargs.get("top_k", 5)
        num_candidates = kwargs.get("num_candidates", 50)

        parsed_query_info = parse_query(query)
        phrases = parsed_query_info.get(PARSE_INFO_KEY_PHRASES, [])
        limitations = parsed_query_info.get(PARSE_INFO_KEY_LIMITATION, [])
        syntax_items = parsed_query_info.get(PARSE_INFO_KEY_SYNTAX, [])

        must_clauses = []
        should_clauses = []
        must_not_clauses = []

        embedding_phrases = []
        if phrases and len(phrases) > 0:
            for phrase in phrases:
                should_clauses.append({"match": {"page_content": phrase}})
                embedding_phrases.append(phrase)
        if limitations and len(limitations) > 0:
            for limitation in limitations:
                # _, value = limitation.split(":", 1)
                # must_clauses.append({"match": {"page_content": value}})
                should_clauses.append({"match": {"page_content": limitation}})
                embedding_phrases.append(limitation)
        if syntax_items and len(syntax_items) > 0:
            for syntax_item in syntax_items:
                if syntax_item.startswith("-"):
                    must_not_clauses.append({"match": {"page_content": syntax_item[1:]}})

        if self.embedding_model is not None:
            embedding_query_str = " ".join(embedding_phrases)
            query_vector = self.embedding_model.embed_query(embedding_query_str)
            knn_query = {
                "knn": {
                    "field": "dense_vector_field",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": num_candidates,
                    "boost": kwargs.get("knn_boost", 1.0)
                }
            }
            should_clauses.append(knn_query)

        query_info = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": should_clauses,
                    "must_not": must_not_clauses,
                }
            },
            "size": top_k
        }
        logger.debug(f"[hybrid_query] Retrieve ElasticSearch by the query info: {query_info}")
        return query_info

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not self.es_client or not self.document_mapper:
            raise ValueError("faulty configuration")  # should not happen

        if self.body_func:
            query_body = self.body_func(query)
        else:
            query_body = self.hybrid_query(query, **self.query_kwargs)
        results = self.es_client.search(index=self.index_name, body=query_body)
        # logger.debug(f"[DefaultElasticsearchRetriever]- Retriever by query: {query} and find results: {results}")
        docs = [self.document_mapper(hit) for hit in results["hits"]["hits"]]
        logger.debug(f"[DefaultElasticsearchRetriever]- Retriever by query: {query} ")
        for doc in docs:
            logger.debug(f"[DefaultElasticsearchRetriever]-Find the doc: {doc}")
        return docs

    def _single_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        content = hit["_source"].pop(self.content_field)
        return Document(page_content=content, metadata=hit)

    def _multi_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        self.content_field = cast(Mapping, self.content_field)
        field = self.content_field[hit["_index"]]
        content = hit["_source"].pop(field)
        return Document(page_content=content, metadata=hit)
