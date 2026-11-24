from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from configs.model_config import logger
from controllers.common.common_constant import MetadataConstants
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.hipporag_with_graph_db import HippoRAG
from core.data_connections.retrievers.storage.documents_storage import BaseDocumentsStorage
from core.data_connections.retrievers.utils.retrieve_util import parse_query


class HippoRetriever(BaseRetriever):
    hippo_rag: HippoRAG
    docstore: BaseDocumentsStorage
    search_kwargs: dict = Field(default_factory=dict)
    enable_logging: bool = True
    kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Parsed query is a list of phrases. Please refer prompt_definitions/builtin/lang/en/tool_definition_prompt.json
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        parsed_query_info = parse_query(query)
        phrases = parsed_query_info.get("phrases", None)
        if phrases is not None and len(phrases) > 0:
            query = phrases

        logger.debug(f"[HippoRetriever] - Query: {query}")
        ranks, scores, logs = self.hippo_rag.rank_docs(query, top_k=self.search_kwargs.get("top_k", 4))
        score_threshold = self.kwargs.get("score_threshold", 0.09)
        rerank_doc_ids = []
        sorted_scores = []
        for i, score in enumerate(scores):
            if score > score_threshold:
                rerank_doc_ids.append(ranks[i])
                sorted_scores.append(score)
        doc_id_to_score = dict(zip(rerank_doc_ids, sorted_scores))
        if len(rerank_doc_ids) == 0:
            logger.warning(f"[HippoRetriever] - No relevant documents found for query: {query} with score threshold: {score_threshold}")
            return []
        docs = self.hippo_rag.get_passage_by_ids(rerank_doc_ids, self.docstore)
        for doc in docs:
            doc.metadata[MetadataConstants.HIPPO_RAG_SCORE] = doc_id_to_score.get(doc.id, 0)
            doc.metadata[MetadataConstants.SCORE] = doc_id_to_score.get(doc.id, 0)
        # sort the docs by the score
        docs = sorted(docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True)
        if self.enable_logging:
            logger.info(f"[HippoRetriever] - Found {len(docs)} documents from docstore for query: {query}")
            if len(docs) == 0:
                logger.warning(f"[HippoRetriever] - No documents found for query: {query} from docstore")
            for i, _ in enumerate(docs):
                logger.debug(f"rank: {i}, score: {docs[i].metadata.get('score', 0.0)}, doc: {docs[i].page_content}")

        return docs