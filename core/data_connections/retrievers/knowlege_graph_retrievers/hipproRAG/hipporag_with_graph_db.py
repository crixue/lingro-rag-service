import json
import os
import pickle
from collections import defaultdict, OrderedDict
from glob import glob
from typing import List, Union

import igraph as ig
import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from scipy.sparse import csr_array
from tqdm import tqdm

from configs.model_config import logger, KB_ROOT_PATH, CACHE_ROOT_PATH
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.named_entity_extraction_parallel import \
    query_ner_named_entity_recognition
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.openAI_compatible_embedding_util import \
    OpenAICompatibleEmbeddingWrapper, normalize_embeddings
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import min_max_normalize, \
    processing_phrases
from core.data_connections.retrievers.storage.documents_storage import BaseDocumentsStorage
from utils.neo4j_client import Neo4jGraphDBClient

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

COLBERT_CKPT_DIR = "exp/colbertv2.0"


class HippoRAG:
    def __init__(self, corpus_name: str = 'hotpotqa',
                 retriever_name: str = None, force_retrieve=False, local_doc_folder_name: str = None,
                 llm_client: BaseChatModel = None, neo4j_client: Neo4jGraphDBClient = None, graph_db_name: str = None,
                 embedding_model: Embeddings = None, user_folder_id: str = None, storage: BaseDocumentsStorage = None,
                 **kwargs):
        self.max_version = 0
        assert retriever_name is not None, 'linking_retriever_name must be provided'
        assert embedding_model is not None, 'embedding_model must be provided'
        if local_doc_folder_name is None:
            local_doc_folder_name = kwargs.get('local_vs_folder_name', None)
        assert local_doc_folder_name is not None, 'local_doc_folder_name/local_vs_folder_name must be provided'
        assert user_folder_id is not None, 'user_folder_id must be provided'
        self.corpus_name = corpus_name
        self.linking_retriever_name = retriever_name
        self.llm_client = llm_client
        self.extraction_type = kwargs.get('extraction_type', 'ner')
        self.graph_type = kwargs.get('graph_type', 'facts_and_sim')
        self.sim_threshold = kwargs.get('sim_threshold', 0.8)
        self.node_specificity = kwargs.get('node_specificity', True)
        self.doc_ensemble = kwargs.get('doc_ensemble', True)
        self.dpr_only = kwargs.get('dpr_only', False)
        self.graph_alg = kwargs.get('graph_alg', 'ppr')
        self.damping = kwargs.get('damping', 0.1)
        self.recognition_threshold = kwargs.get('recognition_threshold', 0.9)
        self.phrase_type = kwargs.get('phrase_type', 'ents_only_lower_preprocess')
        self.output_dir = os.path.join(kwargs.get('kb_output_dir', CACHE_ROOT_PATH) , local_doc_folder_name)
        self.is_chinese_text = kwargs.get('is_chinese_text', True)
        self.storage = storage

        self.neo4j_client = neo4j_client
        self.graph_db_name = graph_db_name
        self.linking_retriever_name_processed = self.linking_retriever_name.replace('/', '_').replace('.', '')
        # self.embed_model = None
        self.embed_model_wrapper = OpenAICompatibleEmbeddingWrapper(embedding_model)
        self.force_retrieve = force_retrieve

        self.phrases = None
        self.kb_node_phrase_embeddings = None
        self.doc_embeddings = None
        self.graph = None
        self.passage_to_phrases_mat = None
        self.kg_inverse_adj_list = None
        self.kg_adj_list = None
        self.passage_id_indices = {}
        self.passage_id_indices_revert = {}
        self.phrase_id_indices = {}
        self.query_result = None
        self.phrases_to_passage_nums_mat = None
        self.statistics = {}
        self.ensembling_debug = []

    def pre_built_retrieve_index(self):
        logger.debug(f"[HippoRAG]-Building graph start...")
        self.retrieve_nodes_info_from_graphdb()

        logger.debug(f"[HippoRAG]-Loading node embeddings start...")
        self.load_node_vectors()

        logger.debug(f"[HippoRAG]-Creating dpr doc embedding start...")
        if self.doc_ensemble:
            self.create_dpr_doc_embedding()

    def query_ner(self, query):
        if self.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities
            try:
                query_ner_json = query_ner_named_entity_recognition(self.llm_client, query, self.is_chinese_text)
                query_ner_list = eval(query_ner_json)['named_entities']

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except Exception as e:
                logger.exception(f'[query_ner] Query NER exception throw ex:{e}')
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector, max_depth=1):
        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.graph.neighbors(node))
                    all_neighborhood.extend(self.graph.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def retrieve_nodes_info_from_graphdb(self):
        retrieve_nodes_info_file_name = 'retrieve_nodes_info'
        possible_phrases_mat_files = glob(
            f'{self.output_dir}/{self.linking_retriever_name_processed}_mean/{retrieve_nodes_info_file_name}_*.p')
        if len(possible_phrases_mat_files) > 0:
            self.max_version = max([int(f.split('_')[-1].split('.')[0]) for f in possible_phrases_mat_files])
        def _load_retrieve_info_from_cache():
            if self.force_retrieve:
                return False
            # doc_to_phrases_mat_{num}.p. num means the version of the doc_to_phrases_mat
            retrieve_nodes_info_load_fp = f'{self.output_dir}/{self.linking_retriever_name_processed}_mean/{retrieve_nodes_info_file_name}_{self.max_version}.p'
            if not os.path.exists(retrieve_nodes_info_load_fp):
                return False
            with open(retrieve_nodes_info_load_fp, 'rb') as f:
                retrieve_info = pickle.load(f)
                self.phrases = retrieve_info['phrases']
                self.kg_adj_list = retrieve_info['kg_adj_list']
                self.kg_inverse_adj_list = retrieve_info['kg_inverse_adj_list']
                self.passage_id_indices_revert = retrieve_info['passage_id_indices_revert']
                self.passage_to_phrases_mat = retrieve_info['passage_to_phrases_mat']
                self.graph = retrieve_info['graph']
            self.phrases_to_passage_nums_mat = self.passage_to_phrases_mat.sum(0).T
            logger.info(f'[retrieve_nodes_info_from_graphdb]-Loaded retrieve nodes info from cache file: {retrieve_nodes_info_load_fp} ')
            return True

        if _load_retrieve_info_from_cache():
            return

        edges = set()
        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)
        kg_adj_id_list = defaultdict()

        self.phrases = list()
        passage_ids = set()

        temp_result = {} #  {phrase_id: (phrase_name, passage_id_list, phrase_weight)}
        phrase_passage_with_weight = {}  # {(phrase_id, passage_id): weight}
        with self.neo4j_client.graph_driver.session(database=self.graph_db_name) as session:
            query = """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                RETURN e1, elementId(e1) as eId1, e2, elementId(e2) as eId2 ,r.weight AS weight
            """
            result = session.run(query)
            for record in result:
                passage1_name = record['e1']['name']
                passage2_name = record['e2']['name']
                if not isinstance(passage1_name, str) or len(passage1_name) == 0 or not isinstance(passage2_name, str) or len(passage2_name) == 0:
                    continue
                weight = record['weight']

                passage_id1_list = record['e1']['passageId']
                passage_ids.update(passage_id1_list)
                passage_id2_list = record['e2']['passageId']
                passage_ids.update(passage_id2_list)

                phrase1_id = record['eId1']
                temp_result[phrase1_id] = (passage1_name, passage_id1_list, weight)
                for passage_id in passage_id1_list:
                    phrase_ref_passage = (phrase1_id, passage_id)
                    if phrase_ref_passage not in phrase_passage_with_weight:
                        phrase_passage_with_weight[phrase_ref_passage] = 1

                phrase2_id = record['eId2']
                temp_result[phrase2_id] = (passage2_name, passage_id2_list, weight)
                for passage_id in passage_id2_list:
                    phrase_ref_passage = (phrase2_id, passage_id)  # e.g. ('4:d50f112b-5ec9-46e4-8fe0-05c8217c2206:15', 'test_extract_doc_4')
                    if phrase_ref_passage not in phrase_passage_with_weight:
                        phrase_passage_with_weight[phrase_ref_passage] = 1

                if (phrase1_id, phrase2_id) not in edges and phrase1_id != phrase2_id:
                    new_graph_plus[(phrase1_id, phrase2_id)] = weight
                    edges.add((phrase1_id, phrase2_id))
                    kg_adj_id_list[(phrase1_id, phrase2_id)] = weight

        sorted_dict = OrderedDict(sorted(temp_result.items()))
        phrase_ids = list()
        for phrase_id, (phrase_name, passage_id_list, phrase_weight) in sorted_dict.items():
            self.phrases.append(phrase_name)
            phrase_ids.append(phrase_id)

        if self.phrases is None or len(self.phrases) == 0:
            logger.warning(f'[retrieve_nodes_info_from_graphdb] - No phrases found in the graph db')
            return

        for key, val in kg_adj_id_list.items():
            phrase1_index = phrase_ids.index(key[0])
            phrase2_index = phrase_ids.index(key[1])
            self.kg_adj_list[phrase1_index][phrase2_index] = val
            self.kg_inverse_adj_list[phrase2_index][phrase1_index] = val

        passage_ids = sorted(list(passage_ids))
        self.passage_id_indices = {passage_id: index for index, passage_id in enumerate(passage_ids)}
        self.passage_id_indices_revert = {index: passage_id for index, passage_id in enumerate(passage_ids)}
        self.phrase_id_indices = {phrase_id: index for index, phrase_id in enumerate(phrase_ids)}
        self.query_result = {}   # {phrase_index: (phrase_id, phrase_name, passage_id_list, phrase_weight)}
        for phrase_id, (phrase_name, passage_id_list, phrase_weight) in temp_result.items():
            phrase_index = self.phrase_id_indices.get(phrase_id)
            passage_id_list = [self.passage_id_indices.get(passage_id) for passage_id in passage_id_list]
            self.query_result[phrase_index] = (phrase_id, phrase_name, passage_id_list, phrase_weight)
        
        passage_to_phrases_dense_matrix = np.zeros((len(passage_ids), len(phrase_ids)))
        for (phrase_id, passage_id), value in phrase_passage_with_weight.items():
            passage_index = self.passage_id_indices.get(passage_id)
            phrase_index = self.phrase_id_indices.get(phrase_id)
            passage_to_phrases_dense_matrix[passage_index, phrase_index] = value

        self.passage_to_phrases_mat = csr_array(passage_to_phrases_dense_matrix, shape=(len(passage_ids), len(phrase_ids)))
        self.phrases_to_passage_nums_mat = self.passage_to_phrases_mat.sum(0).T

        edge_indices = list()  #[phrase1_index, phrase2_index]
        weights = []
        for phrase1_id, phrase2_id in edges:
            edge_indices.append((self.phrase_id_indices.get(phrase1_id), self.phrase_id_indices.get(phrase2_id)))
            weights.append(new_graph_plus[(phrase1_id, phrase2_id)])
        n_vertices = len(phrase_ids)
        self.graph = ig.Graph(n_vertices, edge_indices)
        self.graph.es['weight'] = weights
        logger.info(f'[retrieve_nodes_info_from_graphdb]-Graph built: num vertices: {n_vertices}, num_edges: {len(edge_indices)}')

        retrieve_info = {
            'phrases': self.phrases,
            'kg_adj_list': self.kg_adj_list,
            'kg_inverse_adj_list': self.kg_inverse_adj_list,
            'passage_id_indices_revert': self.passage_id_indices_revert,
            'passage_to_phrases_mat': self.passage_to_phrases_mat,
            'graph': self.graph
        }
        saved_fp = self._save_python_obj_and_housekeeping(retrieve_info, f'{self.output_dir}/{self.linking_retriever_name_processed}_mean',
                                                  retrieve_nodes_info_file_name)
        logger.info(f'[retrieve_nodes_info_from_graphdb]-Saved retrieve nodes info to cache file: {saved_fp}')

    def _save_python_obj_and_housekeeping(self, obj, file_dict, file_name):
        if not os.path.exists(file_dict):
            os.makedirs(file_dict)
        file_path = f'{file_dict}/{file_name}_{self.max_version + 1}.p'
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        housekeeping_file_path = f'{file_dict}/{file_name}_{self.max_version -1}.p'
        if os.path.exists(housekeeping_file_path):
            os.remove(housekeeping_file_path)
        return file_path

    def load_node_vectors(self):
        kb_node_phrase_embeddings_file_prefix_name = 'kb_node_phrase_embeddings'
        def _load_node_vectors_from_cache():
            if self.force_retrieve:
                return False
            kb_node_phrase_embeddings_load_path = \
                f'{self.output_dir}/{self.linking_retriever_name_processed}_mean/{kb_node_phrase_embeddings_file_prefix_name}_{self.max_version}.p'
            if not os.path.exists(kb_node_phrase_embeddings_load_path):
                return False
            with open(kb_node_phrase_embeddings_load_path, 'rb') as f:
                embeddings = pickle.load(f)
                self.kb_node_phrase_embeddings = np.array(embeddings)
            if len(self.kb_node_phrase_embeddings) == 3:
                self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
            logger.info(f'[load_node_vectors]-Loaded phrase embeddings from cache file: {kb_node_phrase_embeddings_load_path}, shape: {self.kb_node_phrase_embeddings.shape}')
            return True
        if _load_node_vectors_from_cache():
            return self.kb_node_phrase_embeddings

        if self.phrases is None or len(self.phrases) == 0:
            logger.warning(f'[load_node_vectors] - No phrases found in the graph db')
            return None
        self.kb_node_phrase_embeddings = self.embed_model_wrapper.encode_text(self.phrases)

        saved_fp = self._save_python_obj_and_housekeeping(self.kb_node_phrase_embeddings,
                                          f'{self.output_dir}/{self.linking_retriever_name_processed}_mean',
                                                  kb_node_phrase_embeddings_file_prefix_name)
        logger.info(f'[load_node_vectors]-Saved phrase embeddings to cache file: {saved_fp}')
        return self.kb_node_phrase_embeddings

    def create_dpr_doc_embedding(self, **kwargs):
        dpr_doc_embedding_file_prefix_name = 'dpr_doc_embedding'
        def _load_dpr_doc_embedding_from_cache():
            if self.force_retrieve:
                return False
            multi_files = glob(f'{self.output_dir}/{self.linking_retriever_name_processed}_mean/{dpr_doc_embedding_file_prefix_name}_*.p')
            max_version = 0
            if len(multi_files) > 0:
                max_version = max([int(f.split('_')[-1].split('.')[0]) for f in multi_files])
            dpr_doc_embedding_load_fp = f'{self.output_dir}/{self.linking_retriever_name_processed}_mean/{dpr_doc_embedding_file_prefix_name}_{max_version}.p'
            if not os.path.exists(dpr_doc_embedding_load_fp):
                return False
            with open(dpr_doc_embedding_load_fp, 'rb') as f:
                embeddings = pickle.load(f)
                self.doc_embeddings = np.array(embeddings)
            logger.info(f'[create_dpr_doc_embedding]-Loaded dpr doc embeddings from cache file: {dpr_doc_embedding_load_fp}, shape: {self.doc_embeddings.shape}')
            return True

        if _load_dpr_doc_embedding_from_cache():
            return self.doc_embeddings
        self.doc_embeddings = np.array([])
        if self.storage is None:
            logger.warning(f'[create_dpr_doc_embedding] - No storage provided and will not create dpr doc embeddings')
            return None
        if self.passage_id_indices_revert is None or len(self.passage_id_indices_revert.keys()) == 0:
            logger.warning(f'[create_dpr_doc_embedding] - No phrases found in the graph db and will not create dpr doc embeddings')
            return None

        sorted_passage_indies = dict(sorted(self.passage_id_indices_revert.items()))
        total_docs = self.storage.retriever_documents(list(sorted_passage_indies.values()))

        # custom_embeddings = []
        # for doc in total_docs:
        #     if doc.page_content and len(doc.page_content) > 0:
        #         embed = self.embed_model_wrapper.encode_text(doc.page_content, norm=False)
        #         custom_embeddings.append(embed)
        #
        # if len(custom_embeddings) > 0:
        #     custom_embeddings = np.squeeze(np.array(custom_embeddings), axis=1)
        #     custom_embeddings = np.array(normalize_embeddings(custom_embeddings))

        mini_batch_count = 4
        mini_batch_docs = [total_docs[i:i + mini_batch_count] for i in range(0, len(total_docs), mini_batch_count)]
        for docs in mini_batch_docs:
            if len(docs) == 0:
                continue

            embed = self.embed_model_wrapper.encode_text([doc.page_content for doc in docs],
                                                 return_numpy=True)
            self.doc_embeddings = np.concatenate((self.doc_embeddings, embed), axis=0) if len(self.doc_embeddings) > 0 else embed

        if len(self.doc_embeddings) == 3:
            self.doc_embeddings = np.squeeze(self.doc_embeddings, axis=1)

        self.doc_embeddings = normalize_embeddings(self.doc_embeddings)
        # self.doc_embeddings = custom_embeddings
        saved_fp = self._save_python_obj_and_housekeeping(self.doc_embeddings,
                                          f'{self.output_dir}/{self.linking_retriever_name_processed}_mean',
                                                  dpr_doc_embedding_file_prefix_name)
        logger.info(f"[create_index_and_save_to_storage]-Creating dpr doc embedding, total docs: {len(total_docs)} and saved to {saved_fp}")
        return self.doc_embeddings

    def rank_docs(self, query: Union[str, list[str]], top_k=10) -> tuple[List, List[float], dict]:
        """
        Rank documents based on the query
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores, logs
        """
        self.pre_built_retrieve_index()

        if isinstance(query, str):
            query_ner_list = self.query_ner(query)
        else:
            query_ner_list = query

        all_phrase_weights = np.zeros(len(self.phrases))
        linking_score_map = {}
        if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
            all_phrase_weights, linking_score_map = self.link_node_by_dpr(query_ner_list)
        else:
            logger.warning(f'[rank_docs] - No entities found in the query: {query} !')

        # Run Personalized PageRank (PPR) or other Graph Algorithm Doc Scores
        ppr_doc_prob = None
        ppr_phrase_probs = None
        if not self.dpr_only:
            if len(query_ner_list) > 0:
                combined_vector = np.max([all_phrase_weights], axis=0)

                if self.graph_alg == 'ppr':
                    ppr_phrase_probs = self.run_pagerank_igraph_chunk([all_phrase_weights])[0]  # (num phrases, 1)
                elif self.graph_alg == 'none':
                    ppr_phrase_probs = combined_vector
                elif self.graph_alg == 'neighbor_2':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 2)
                elif self.graph_alg == 'neighbor_3':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                elif self.graph_alg == 'paths':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                else:
                    assert False, f'Graph Algorithm {self.graph_alg} Not Implemented'

                # fact_prob = self.facts_to_phrases_mat.dot(ppr_phrase_probs)  # (num facts, 1)
                # ppr_doc_prob = docs_to_facts_mat.dot(fact_prob)  # (num docs, 1)
                # ppr_doc_prob_maxtrix = self.passage_to_phrases_dense_matrix.dot(ppr_phrase_probs)
                ppr_doc_prob = self.passage_to_phrases_mat.dot(ppr_phrase_probs)  # (num docs, 1)
                ppr_doc_prob = min_max_normalize(ppr_doc_prob)
            else:  # dpr_only or no entities found
                docs_num = self.passage_to_phrases_mat.shape[1]
                ppr_doc_prob = np.ones(docs_num) / docs_num

        # Combine Query-Doc and PPR Scores
        self.doc_ensemble = self.doc_ensemble and self.doc_embeddings is not None
        if self.doc_ensemble or self.dpr_only:
            if len(linking_score_map.values()) > 0 and np.min(
                    list(linking_score_map.values())) > self.recognition_threshold:  # high confidence in named entities
                doc_prob = ppr_doc_prob
                self.statistics['ppr'] = self.statistics.get('ppr', 0) + 1
            else:
                # Get Query Doc Scores
                query_embedding = self.embed_model_wrapper.encode_text(query)
                query_doc_scores = np.dot(self.doc_embeddings, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]

                # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
                if len(query_ner_list) == 0:
                    doc_prob = query_doc_scores
                    self.statistics['doc'] = self.statistics.get('doc', 0) + 1
                else:  # relatively low confidence in named entities, combine the two scores
                    # the higher threshold, the higher chance to use the doc ensemble
                    query_doc_scores = min_max_normalize(query_doc_scores)
                    doc_prob = ppr_doc_prob * 0.5 + query_doc_scores * 0.5

                    top_ppr = np.argsort(ppr_doc_prob)[::-1][:10]
                    top_ppr = [(top, ppr_doc_prob[top]) for top in top_ppr]

                    top_doc = np.argsort(query_doc_scores)[::-1][:10]
                    top_doc = [(top, query_doc_scores[top]) for top in top_doc]

                    top_hybrid = np.argsort(doc_prob)[::-1][:10]
                    top_hybrid = [(top, doc_prob[top]) for top in top_hybrid]

                    self.ensembling_debug.append((top_ppr, top_doc, top_hybrid))
                    self.statistics['ppr_doc_ensemble'] = self.statistics.get('ppr_doc_ensemble', 0) + 1
        else:
            doc_prob = ppr_doc_prob

        # Return ranked docs and ranked scores
        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]

        if not (self.dpr_only) and len(query_ner_list) > 0:
            # logs
            phrase_one_hop_triples = []
            for phrase_id in np.where(all_phrase_weights > 0)[0]:
                # get all the triples that contain the phrase from self.graph_plus
                for t in list(self.kg_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1]])
                for t in list(self.kg_inverse_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1], 'inv'])

            # get top ranked nodes from doc_prob and self.doc_to_phrases_mat
            nodes_in_retrieved_doc = []
            for doc_id in sorted_doc_ids[:5]:
                node_id_in_doc = list(np.where(self.passage_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
                nodes_in_retrieved_doc.append([self.phrases[node_id] for node_id in node_id_in_doc])

            # get top ppr_phrase_probs
            top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]

            # get phrases for top_pagerank_phrase_ids
            top_ranked_nodes = [self.phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]
            logs = {'named_entities': query_ner_list,
                    'linked_node_scores': [list(k) + [float(v)] for k, v in linking_score_map.items()],
                    '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                    'top_ranked_nodes': top_ranked_nodes, 'nodes_in_retrieved_doc': nodes_in_retrieved_doc}
        else:
            logs = {}

        top_passage_indies = sorted_doc_ids.tolist()[:top_k]
        passage_ids= [self.passage_id_indices_revert[passage_index] for passage_index in top_passage_indies]
        return passage_ids, sorted_scores.tolist()[:top_k], logs

    def run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run pagerank on the graph
        :param reset_prob_chunk:
        :return: PageRank probabilities
        """
        pageranked_probabilities = []

        for reset_prob in tqdm(reset_prob_chunk, desc='pagerank chunk'):
            pageranked_probs = self.graph.personalized_pagerank(vertices=range(len(self.phrases)),
                                                                damping=self.damping, directed=False,
                                                                weights='weight', reset=reset_prob, implementation='prpack')
            pageranked_probabilities.append(np.array(pageranked_probs))

        return np.array(pageranked_probabilities)

    def link_node_by_dpr(self, query_ner_list: list) -> tuple[np.ndarray, dict]:
        """
        Get the most similar phrases (as vector) in the KG given the named entities
        :param query_ner_list:
        :return:
        """
        query_ner_embeddings = self.embed_model_wrapper.encode_text(query_ner_list)
        # Get Closest Entity Nodes
        prob_vectors = np.dot(query_ner_embeddings,
                              self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_phrase_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            phrase_index = np.argmax(prob_vector)  # the phrase with the highest similarity
            linked_phrase_ids.append(phrase_index)
            max_scores.append(prob_vector[phrase_index])

        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_phrase_weights = np.zeros(len(self.phrases))  # (num_phrases)

        for phrase_index in linked_phrase_ids:
            if self.node_specificity:
                if self.phrases_to_passage_nums_mat[phrase_index] == 0:  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.phrases_to_passage_nums_mat[phrase_index]

                all_phrase_weights[phrase_index] = weight
            else:
                all_phrase_weights[phrase_index] = 1.0

        linking_score_map = {(query_phrase, self.phrases[linked_phrase_id]): max_score
                             for linked_phrase_id, max_score, query_phrase in
                             zip(linked_phrase_ids, max_scores, query_ner_list)}
        return all_phrase_weights, linking_score_map

    def get_passage_by_ids(self, passage_ids: list = None, storage: BaseDocumentsStorage = None, ) -> List[Document]:
        if passage_ids is None or len(passage_ids) == 0 or storage is None:
            return []
        return storage.retriever_documents(passage_ids)