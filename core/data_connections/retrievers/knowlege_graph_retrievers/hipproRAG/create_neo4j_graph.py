import json
import os
import pickle
from glob import glob

import numpy as np
from pandas._typing import Self
from pydantic import BaseModel, model_validator
from scipy.sparse import csr_array
from tqdm import tqdm

from configs.model_config import logger
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import EXTRACTED_ENTITIES_KEY, \
    EXTRACTED_TRIPLES_KEY, PASSAGE_KEY
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import processing_phrases
from utils.neo4j_client import Neo4jGraphDBClient

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'


class HippoRAGGraphCreator(BaseModel):
    version: str = 'v3'
    inter_triple_weight: float = 1.0
    similarity_max: float = 1.0
    dataset: str
    extraction_type: str
    extraction_model: str
    retriever_name: str
    processed_retriever_name: str
    neo4j_client: Neo4jGraphDBClient = None
    threshold: float = 0.8
    create_graph_flag: bool = False
    cosine_sim_edges: bool = False
    graph_db_name: str = 'hipporag'
    extended_graph_label_info: dict = None
    output_dir: str = 'output'
    extracted_triples: list = []
    phrase_type: str = 'ents_only_lower_preprocess'
    graph_type: str = 'facts'

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def validate_extraction_model(self) -> Self:
        possible_files = glob(
            '{}/openie_{}_results_{}_{}_*.json'.format(self.output_dir, self.dataset, self.extraction_type,
                                                       self.extraction_model))
        max_samples = np.max(
            [int(file.split('{}_'.format(self.extraction_model))[1].split('.json')[0]) for file in possible_files])
        with open(
                f'{self.output_dir}/openie_{self.dataset}_results_{self.extraction_type}_{self.extraction_model}_{max_samples}.json',
                'r') as json_file:
            extracted_file = json.load(json_file)
        self.extracted_triples = extracted_file['docs']
        if self.extraction_model != 'gpt-3.5-turbo-1106':
            self.extraction_type = f'{self.extraction_type}_{self.extraction_model}'
        if self.cosine_sim_edges:
            self.graph_type = 'facts_and_sim'
        return self

    def creat_neo4j_graph(self):
        phrases = []
        entities = []
        relations = {}
        incorrectly_formatted_triples = []
        triples_wo_ner_entity = []
        triple_tuples = []
        full_neighborhoods = {}
        graph_driver = self.neo4j_client.graph_driver
        for i, row in tqdm(enumerate(self.extracted_triples), total=len(self.extracted_triples)):
            raw_ner_entities = row[EXTRACTED_ENTITIES_KEY]
            ner_entities = [processing_phrases(p) for p in row[EXTRACTED_ENTITIES_KEY]]

            triples = row[EXTRACTED_TRIPLES_KEY]

            doc_json = row

            clean_triples = []
            unclean_triples = []
            doc_entities = set()

            # Populate Triples from OpenIE
            for triple in triples:

                triple = [str(s) for s in triple]

                if len(triple) > 1:
                    if len(triple) != 3:
                        clean_triple = [processing_phrases(p) for p in triple]

                        incorrectly_formatted_triples.append(triple)
                        unclean_triples.append(triple)
                    else:
                        clean_triple = [processing_phrases(p) for p in triple]

                        clean_triples.append(clean_triple)
                        phrases.extend(clean_triple)

                        head_ent = clean_triple[0]
                        tail_ent = clean_triple[2]

                        if head_ent not in ner_entities and tail_ent not in ner_entities:
                            triples_wo_ner_entity.append(triple)

                        relations[(head_ent, tail_ent)] = clean_triple[1]

                        raw_head_ent = triple[0]
                        raw_tail_ent = triple[2]

                        entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                        entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                        full_neighborhoods[raw_head_ent] = entity_neighborhood

                        entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                        entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                        full_neighborhoods[raw_tail_ent] = entity_neighborhood

                        for triple_entity in [clean_triple[0], clean_triple[2]]:
                            entities.append(triple_entity)
                            doc_entities.add(triple_entity)

            doc_json['entities'] = list(set(doc_entities))
            doc_json['clean_triples'] = clean_triples
            doc_json['noisy_triples'] = unclean_triples
            triple_tuples.append(clean_triples)

        with graph_driver.session() as session:
            session.run(f"CREATE DATABASE {self.graph_db_name} IF NOT EXISTS")

        with graph_driver.session(database=self.graph_db_name) as session:
            for triple in clean_triples:
                query = """
                    MERGE (e1:Entity {name: $head})
                    MERGE (e2:Entity {name: $tail})
                    MERGE (e1)-[r:""" + str(triple[1]) + """ {name: $relation}]-(e2)
                    ON CREATE SET r.occurrences = 1, e1.dataSource = apoc.coll.toSet([$dataset]), e2.dataSource = apoc.coll.toSet([$dataset]),
                        e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                    ON MATCH SET r.occurrences = r.occurrences + 1,
                        e1.dataSource = apoc.coll.toSet(e1.dataSource + $dataset),
                        e2.dataSource = apoc.coll.toSet(e2.dataSource + $dataset),
                        r.dataSource = apoc.coll.toSet(r.dataSource + $dataset),
                        e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                """
                session.run(query, parameters={'head': triple[0], 'relation': triple[1], 'tail': triple[2],
                                               'dataset': self.dataset})
                if self.extended_graph_label_info is not None:
                    var1 = """
                        MATCH (e1: Entity {name: $head})-[r:""" + str(triple[1]) + """ {name: $relation}]-(e2: Entity {name: $tail})
                    """
                    for key, val in self.extended_graph_label_info.items():
                        escaped_label = key.replace("\\u0060", "`").replace("`", "``")
                        var1 += f"ON CREATE SET e1.{escaped_label} = {val}, e2.{escaped_label} = {val}\n"
                        var1 += f"ON MATCH SET e1.{escaped_label} = {val}, e2.{escaped_label} = {val}\n"
                    session.run(query, parameters={'head': triple[0], 'relation': triple[1], 'tail': triple[2]})

            # passage_json.append(doc_json)
        logger.info('[create_graph]-Number of Incorrectly Formatted Triples (ChatGPT Error): {}'.format(
            len(incorrectly_formatted_triples)))
        logger.info(
            '[create_graph]-Number of Triples w/o NER Entities (ChatGPT Error): {}'.format(len(triples_wo_ner_entity)))

    def create_full_graph_and_phrase_matrix(self, triple_tuples: list, unique_phrases: list):
        '''
        使用的时候再运算
        '''
        inter_triple_weight = 1.0
        # node_json = [{'idx': i, 'name': p} for i, p in enumerate(unique_phrases)]
        # kb_phrase_df = pd.DataFrame(unique_phrases)
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}

        lose_facts = []

        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])

        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}
        # fact_json = [{'idx': i, 'head': t[0], 'relation': t[1], 'tail': t[2]} for i, t in enumerate(lose_facts)]

        with open('output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                            self.extraction_type, self.version), 'wb') as f:
            pickle.dump(kb_phrase_dict, f)
        with open('output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                          self.extraction_type, self.version), 'wb') as f:
            pickle.dump(lose_fact_dict, f)

        graph_json = {}

        docs_to_facts = {}  # (Num Docs, Num Facts)
        facts_to_phrases = {}  # (Num Facts, Num Phrases)
        graph = {}  # (Num Phrases, Num Phrases)

        num_triple_edges = 0
        num_doc_edges = 0
        with self.graph_driver.session(database=self.graph_db_name) as session:
            for doc_id, triples in tqdm(enumerate(triple_tuples), total=len(triple_tuples)):
                doc_phrases = []
                fact_edges = []

                # Iterate over triples
                for triple in triples:
                    triple = tuple(triple)

                    fact_id = lose_fact_dict[triple]

                    if len(triple) == 3:
                        relation = triple[1]
                        triple = np.array(triple)[[0, 2]]

                        docs_to_facts[(doc_id, fact_id)] = 1

                        for i, phrase in enumerate(triple):
                            phrase_id = kb_phrase_dict[phrase]
                            doc_phrases.append(phrase_id)

                            facts_to_phrases[(fact_id, phrase_id)] = 1

                            for phrase2 in triple[i + 1:]:
                                phrase2_id = kb_phrase_dict[phrase2]

                                fact_edge_r = (phrase_id, phrase2_id)
                                fact_edge_l = (phrase2_id, phrase_id)

                                fact_edges.append(fact_edge_r)
                                fact_edges.append(fact_edge_l)

                                graph[fact_edge_r] = graph.get(fact_edge_r, 0.0) + inter_triple_weight
                                graph[fact_edge_l] = graph.get(fact_edge_l, 0.0) + inter_triple_weight

                                phrase_edges = graph_json.get(phrase, {})
                                edge = phrase_edges.get(phrase2, ('triple', 0))
                                phrase_edges[phrase2] = ('triple', edge[1] + 1)
                                graph_json[phrase] = phrase_edges

                                phrase_edges = graph_json.get(phrase2, {})
                                edge = phrase_edges.get(phrase, ('triple', 0))
                                phrase_edges[phrase] = ('triple', edge[1] + 1)
                                graph_json[phrase2] = phrase_edges

                                num_triple_edges += 1

                                query = """
                                    MERGE (e1:Entity {name: $head})
                                    MERGE (e2:Entity {name: $tail})
                                    MERGE (e1)-[r:""" + str(relation) + """ {name: $relation}]-(e2)
                                    ON CREATE SET r.weight = $weight,
                                        e1.dataSource = apoc.coll.toSet([$dataset]), e2.dataSource = apoc.coll.toSet([$dataset]),
                                        e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                                    ON MATCH SET r.weight = r.weight + $weight,
                                        e1.dataSource = apoc.coll.toSet(e1.dataSource + $dataset),
                                        e2.dataSource = apoc.coll.toSet(e2.dataSource + $dataset),
                                        r.dataSource = apoc.coll.toSet(r.dataSource + $dataset),
                                        e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                                """
                                session.run(query,
                                            parameters={'head': phrase, 'relation': relation, 'tail': phrase2,
                                                        'dataset': self.dataset, 'weight': inter_triple_weight})

        with open('output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                             self.extraction_type, self.version), 'wb') as f:
            pickle.dump(docs_to_facts, f)
        with open('output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                                 self.extraction_type, self.version), 'wb') as f:
            pickle.dump(facts_to_phrases, f)

        docs_to_facts_mat = csr_array(([int(v) for v in docs_to_facts.values()],
                                       ([int(e[0]) for e in docs_to_facts.keys()],
                                        [int(e[1]) for e in docs_to_facts.keys()])),
                                      shape=(len(triple_tuples), len(lose_facts)))  # (Num Docs, Num Facts)
        facts_to_phrases_mat = csr_array(([int(v) for v in facts_to_phrases.values()],
                                          ([e[0] for e in facts_to_phrases.keys()],
                                           [e[1] for e in facts_to_phrases.keys()])),
                                         shape=(len(lose_facts), len(unique_phrases)))  # (Num Facts, Num Phrases)

        with open('output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                                 self.extraction_type, self.version), 'wb') as f:
            pickle.dump(docs_to_facts_mat, f)
        with open('output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(self.dataset, self.graph_type, self.phrase_type,
                                                                                     self.extraction_type, self.version),'wb') as f:
            pickle.dump(facts_to_phrases_mat, f)
