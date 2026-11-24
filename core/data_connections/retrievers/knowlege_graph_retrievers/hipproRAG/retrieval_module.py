import copy
import gc
import os
import os.path
import pickle
from glob import glob

import faiss
import numpy as np
import pandas as pd
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from configs.model_config import logger, EMBEDDING_MINI_BATCH_SIZE
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.openAI_compatible_embedding_util import \
    OpenAICompatibleEmbeddingWrapper, normalize_embeddings
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import processing_phrases, \
    mean_pooling, mean_pooling_embedding_v2
from utils.neo4j_client import Neo4jGraphDBClient


class RetrievalModule:
    """
    Class designed to retrieve potential synonymy candidates for a set of UMLS terms from a set of entities.
    """

    def _get_exists_entities_relations_from_graphdb(self, string_filename='kb_to_kb'):
        with self.neo4j_client.driver.session(database=self.graph_db_name) as session:
            if string_filename.startswith('kb_to_kb'):
                entities_result = session.run("MATCH (e:Entity) RETURN e.name AS name")
                entities = [record["name"] for record in entities_result]
                return entities
            elif string_filename.startswith('rel_kb_to_kb'):
                relations_result = session.run(
                    "MATCH (e1:Entity)-[r]->(e2:Entity) RETURN DISTINCT (r.name) AS relation")
                relations = [record["relation"] for record in relations_result]
                return relations

    def _get_unique_entities(self):
        unique_entities = set()
        with self.neo4j_client.driver.session(database=self.graph_db_name) as session:
            entities_result = session.run("MATCH (e:Entity) RETURN e")
            for record in entities_result:
                unique_entities.add(record["e"])
            return list(unique_entities)

    def _get_AUI_strings_from_graphdb(self, string_filename):
        if string_filename.startswith('kb_to_kb'):
            entities = self._get_exists_entities_relations_from_graphdb(string_filename)
            unique_phrases = list(np.unique(entities))
            kb = pd.DataFrame(unique_phrases, columns=['strings'])
            kb2 = copy.deepcopy(kb)
            kb['type'] = 'query'
            kb2['type'] = 'kb'
            kb_full = pd.concat([kb, kb2])
            return kb_full
        elif string_filename.startswith('rel_kb_to_kb'):
            relations = self._get_exists_entities_relations_from_graphdb(string_filename)
            unique_relations = np.unique(list(relations) + ['equivalent'])
            rel_kb = pd.DataFrame(unique_relations, columns=['strings'])
            rel_kb2 = copy.deepcopy(rel_kb)
            rel_kb['type'] = 'query'
            rel_kb2['type'] = 'kb'
            rel_kb_full = pd.concat([rel_kb, rel_kb2])
            return rel_kb_full

    def __init__(self,
                 retriever_name,
                 pool_method='mean',
                 pretrained_model_path: str | None = None,
                 device='cuda',
                 embedding_model: Embeddings = None,
                 neo4j_client: Neo4jGraphDBClient = None,
                 graph_db_name: str = 'hipporag',
                 vector_dir: str = 'data/lm_vectors',
                 ):
        """
        Args:
            retriever_name: Retrieval names can be one of 3 types
                2) The name of a pickle file mapping AUIs to precomputed vectors
                3) A huggingface transformer model
        """

        self.pool_method = pool_method
        self.neo4j_client = neo4j_client
        self.graph_db_name = graph_db_name

        self.retriever_name = retriever_name
        self.pretrained_model_path = pretrained_model_path

        # self.retrieval_name_dir = None
        self.device = device

        # Search for pickle file
        logger.info('[RetrievalModule]-No Pre-Computed Vectors. Confirming PLM Model.')

        if embedding_model is not None:
            self.embedding_model_wrapper = OpenAICompatibleEmbeddingWrapper(embedding_model)
        else:
            try:
                if 'ckpt' in retriever_name:
                    if pretrained_model_path is not None:
                        self.plm = AutoModel.load_from_checkpoint(pretrained_model_path)
                    else:
                        self.plm = AutoModel.load_from_checkpoint(retriever_name)
                else:
                    if pretrained_model_path is not None:
                        self.plm = AutoModel.from_pretrained(pretrained_model_path)
                    else:
                        self.plm = AutoModel.from_pretrained(retriever_name)
            except:
                assert False, print('{} is an invalid retriever name. Check Documentation.'.format(retriever_name))

        # If not pre-computed, create vectors
        # self.retrieval_name_dir = vector_dir + '/' + self.retriever_name.replace('/', '_').replace('.',
        #                                                                                            '') + '_' + pool_method
        # if not (os.path.exists(self.retrieval_name_dir)):
        #     os.makedirs(self.retrieval_name_dir)

    def retrieve_neighbours(self, string_filename):
        # Get AUI Strings to be Encoded
        string_df = self._get_AUI_strings_from_graphdb(string_filename)
        if string_df.shape[0] == 0:
            logger.error(f'[RetrievalModule]-No Strings Found in {string_filename}.')
            raise ValueError(f'[RetrievalModule]-No Strings Found in {string_filename}.')
        string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
        sorted_df = self.create_sorted_df(string_df.strings.values)

        # Get previously computed vectors
        # precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors(self.retrieval_name_dir)
        precomp_strings = []
        precomp_vectors = []
        # Identify Missing Strings
        missing_strings = self.find_missing_strings(sorted_df.strings.unique(), precomp_strings)

        # Encode Missing Strings
        if len(missing_strings) > 0:
            logger.info(f'[RetrievalModule]-Encoding {len(missing_strings)} Missing Strings')
            # new_vectors, new_strings, = self.encode_strings(missing_strings, pool_method)
            new_vectors, new_strings, = self.encode_strings_v2(missing_strings, self.pool_method)

            precomp_strings = list(precomp_strings)
            precomp_vectors = list(precomp_vectors)

            precomp_strings.extend(list(new_strings))
            precomp_vectors.extend(list(new_vectors))

            precomp_vectors = np.array(precomp_vectors)
            # self.save_vecs(precomp_strings, precomp_vectors, self.retrieval_name_dir)

        self.vector_dict = self.make_dictionary(sorted_df, precomp_strings, precomp_vectors)

        logger.info('[RetrievalModule]-Vectors Loaded.')
        queries = string_df[string_df.type == 'query']
        kb = string_df[string_df.type == 'kb']

        nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values, k=2047)
        # save_path = self.retrieval_name_dir + '/nearest_neighbor_{}.p'.format(string_filename)
        # with open(save_path, 'wb') as f:
        #     pickle.dump(nearest_neighbors, f)
        #     logger.info(f'[RetrievalModule]-Nearest Neighbors have saved to {save_path}')
        return nearest_neighbors

    def _isdigit(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def augment_graphdb_with_similarity(self, string_filename="kb_to_kb", threshold=0.80, similarity_pick_nums: int = 20,):
        nearest_neighbors = self.retrieve_neighbours(string_filename)
        kb_similarity = {processing_phrases(k): v for k, v in nearest_neighbors.items()}
        entities = self._get_unique_entities()
        kb_phrase_dict = {processing_phrases(str(k['name'])): k.element_id for k in entities}

        similarity_max = 1.0
        graph_plus = {}

        for phrase in tqdm(kb_similarity.keys(), total=len(kb_similarity), desc='[augment_graphdb_with_similarity]-Augmenting GraphDB with Similarity'):
            synonyms = []

            var1 = processing_phrases(phrase)
            if len(var1) < 2 or self._isdigit(var1):
                continue

            phrase_id = kb_phrase_dict.get(phrase, None)

            if phrase_id is not None:
                nns = kb_similarity[phrase]
                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    nn = processing_phrases(nn)
                    if len(nn) < 2 or self._isdigit(nn):
                        continue
                    if score < threshold or num_nns > similarity_pick_nums:
                        break

                    if nn != phrase:
                        phrase2_id = kb_phrase_dict.get(nn)
                        if phrase2_id is not None:
                            sim_edge = (phrase_id, phrase2_id)
                            synonyms.append((nn, score))

                            # relations[(phrase, phrase2_id)] = 'equivalent'
                            graph_plus[sim_edge] = similarity_max * score
                            num_nns += 1

        import logging
        # Set the logging level for the Neo4j driver to WARNING
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        with self.neo4j_client.driver.session(database=self.graph_db_name) as session:
            for (phrase1_id, phrase2_id), score in graph_plus.items():
                query = """
                    MATCH (e1:Entity)
                    MATCH (e2:Entity)
                    WHERE elementId(e1) = $phrase1
                      AND elementId(e2) = $phrase2
                      AND NOT (e1)-->(e2)
                    WITH e1, e2
                    MERGE (e1)-[r:equivalent {name: 'equivalent'}]->(e2)
                    ON CREATE SET r.weight = $score, r.updatedAt = datetime()
                    ON MATCH SET r.weight = $score, r.updatedAt = datetime()
                """
                session.run(query, phrase1=phrase1_id, phrase2=phrase2_id, score=score)

    def get_precomputed_plm_vectors(self, retrieval_name_dir):

        # Load or Create a DataFrame sorted by phrase length for efficient PLM computation
        strings = self.load_precomp_strings(retrieval_name_dir)
        vectors = self.load_plm_vectors(retrieval_name_dir)

        return strings, vectors

    def create_sorted_df(self, strings):
        lengths = []

        for string in tqdm(strings):
            lengths.append(len(str(string)))

        lengths_df = pd.DataFrame(lengths)
        lengths_df['strings'] = strings

        return lengths_df.sort_values(0)

    def save_vecs(self, strings, vectors, direc_name, bin_size=50000):
        with open(direc_name + '/encoded_strings.txt', 'w') as f:
            for string in strings:
                f.write(string + '\n')

        split_vecs = np.array_split(vectors, int(len(vectors) / bin_size) + 1)

        for i, vecs in tqdm(enumerate(split_vecs)):
            with open(direc_name + '/vecs_{}.p'.format(i), 'wb') as f:
                pickle.dump(vecs, f)

    def load_precomp_strings(self, retrieval_name_dir):
        filename = retrieval_name_dir + '/encoded_strings.txt'

        if not (os.path.exists(filename)):
            return []

        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

        return lines

    def load_plm_vectors(self, retrieval_name_dir):
        vectors = []

        logger.info(f'[RetrievalModule]-Loading all PLM Vectors from {retrieval_name_dir}/vecs_*.p')
        files = glob(retrieval_name_dir + '/vecs_*.p')

        if len(files) == 0:
            return vectors

        for i in tqdm(range(len(files))):
            i_files = glob(retrieval_name_dir + '/*_{}.p'.format(i))
            if len(i_files) != 1:
                break
            else:
                with open(i_files[0], 'rb') as f:
                    vectors.append(pickle.load(f))

        vectors = np.vstack(vectors)

        return vectors

    def find_missing_strings(self, relevant_strings, precomputed_strings):

        return list(set(relevant_strings).difference(set(precomputed_strings)))

    def make_dictionary(self, sorted_df, precomp_strings, precomp_vectors):
        precomp_string_ids = {}

        for i, string in enumerate(precomp_strings):
            precomp_string_ids[string] = i

        vector_dict = {}

        for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
            string = row.strings

            try:
                vector_id = precomp_string_ids[string]
                vector_dict[string] = precomp_vectors[vector_id]
            except:
                logger.exception(f'[RetrievalModule]-String {string} not found in precomputed vectors.')

        return vector_dict

    def encode_strings_v2(self, strs_to_encode: list[str], pool_method='mean'):
        all_cls = []
        all_strings = []

        mini_batch_size = EMBEDDING_MINI_BATCH_SIZE
        batch_encoding_strings = [strs_to_encode[i:i + mini_batch_size] for i in range(0, len(strs_to_encode), mini_batch_size)]

        for mini_batch in tqdm(batch_encoding_strings, total=len(batch_encoding_strings), desc='Encoding Strings'):
            embeddings = self.embedding_model_wrapper.encode_text(mini_batch)
            all_cls.append(embeddings)
            all_strings.extend(mini_batch)

        all_cls = np.vstack(all_cls)
        all_cls = normalize_embeddings(all_cls)

        assert len(all_cls) == len(all_strings)
        assert all([all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))])

        return all_cls, all_strings

    def encode_strings(self, strs_to_encode, pool_method):
        self.plm.to('cuda')
        if self.pretrained_model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)

        # Sorting Strings by length
        sorted_missing_strings = [len(s) for s in strs_to_encode]
        strs_to_encode = list(np.array(strs_to_encode)[np.argsort(sorted_missing_strings)])

        all_cls = []
        all_strings = []
        num_strings_proc = 0

        with torch.no_grad():

            batch_sizes = []

            text_batch = []
            max_pad_size = 0

            for i, string in tqdm(enumerate(strs_to_encode), total=len(strs_to_encode)):

                length = len(tokenizer.tokenize(string))

                text_batch.append(string)
                num_strings_proc += 1

                if length > max_pad_size:
                    max_pad_size = length

                if max_pad_size * len(text_batch) > 50000 or num_strings_proc == len(strs_to_encode):

                    text_batch = list(text_batch)
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.plm.config.max_length)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    input_ids = input_ids.to('cuda')
                    attention_mask = attention_mask.to('cuda')

                    outputs = self.plm(input_ids, attention_mask=attention_mask)

                    if pool_method == 'cls':
                        embeddings = outputs[0][:, 0, :]

                    elif pool_method == 'mean':
                        embeddings = mean_pooling(outputs[0], attention_mask)

                    all_cls.append(embeddings.cpu().numpy())
                    all_strings.extend(text_batch)

                    batch_sizes.append(len(text_batch))

                    text_batch = []
                    max_pad_size = 0

        all_cls = np.vstack(all_cls)

        assert len(all_cls) == len(all_strings)
        assert all([all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))])

        return all_cls, all_strings

    def retrieve_knn(self, queries, knowledge_base, k=2047):

        original_vecs = []
        new_vecs = []

        for string in knowledge_base:
            original_vecs.append(self.vector_dict[string])

        for string in queries:
            new_vecs.append(self.vector_dict[string])

        if len(original_vecs) == 0 or len(new_vecs) == 0:
            return {}

        original_vecs = np.vstack(original_vecs)
        new_vecs = np.vstack(new_vecs)

        original_vecs = original_vecs.astype(np.float32)
        new_vecs = new_vecs.astype(np.float32)

        faiss.normalize_L2(original_vecs)
        faiss.normalize_L2(new_vecs)

        # Preparing Data for k-NN Algorithm
        dim = len(original_vecs[0])
        index_split = 4
        index_chunks = np.array_split(original_vecs, index_split)
        query_chunks = np.array_split(new_vecs, 100)

        # Building and Querying FAISS Index by parts to keep memory usage manageable.
        index_chunk_D = []
        index_chunk_I = []

        current_zero_index = 0

        for num, index_chunk in enumerate(index_chunks):
            index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)  # build the index

            if faiss.get_num_gpus() > 1:
                gpu_resources = []

                for i in range(faiss.get_num_gpus()):
                    res = faiss.StandardGpuResources()
                    gpu_resources.append(res)

                gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
            else:
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

            gpu_index.add(index_chunk)

            D, I = [], []

            for q in tqdm(query_chunks, desc='[retrieve_knn]-Querying FAISS Index'):
                d, i = gpu_index.search(q, k)

                i += current_zero_index

                D.append(d)
                I.append(i)

            index_chunk_D.append(D)
            index_chunk_I.append(I)

            current_zero_index += len(index_chunk)

            #             print(subprocess.check_output(['nvidia-smi']))

            del gpu_index
            del gpu_resources
            gc.collect()

        stacked_D = []
        stacked_I = []

        for D, I in zip(index_chunk_D, index_chunk_I):
            D = np.vstack(D)
            I = np.vstack(I)

            stacked_D.append(D)
            stacked_I.append(I)

        del index_chunk_D
        del index_chunk_I
        gc.collect()

        stacked_D = np.hstack(stacked_D)
        stacked_I = np.hstack(stacked_I)

        full_sort_I = []
        full_sort_D = []

        for d, i in tqdm(zip(stacked_D, stacked_I)):
            sort_indices = np.argsort(d, kind='stable')

            sort_indices = sort_indices[::-1]

            i = i[sort_indices][:k]
            d = d[sort_indices][:k]

            full_sort_I.append(i)
            full_sort_D.append(d)

        del stacked_D
        del stacked_I
        gc.collect()

        sorted_candidate_dictionary = {}

        for new_index, nn_info in tqdm(enumerate(zip(full_sort_I, full_sort_D))):
            nn_inds, nn_dists = nn_info
            nns = [knowledge_base[i] for i in nn_inds]

            sorted_candidate_dictionary[queries[new_index]] = (nns, nn_dists)

        return sorted_candidate_dictionary
