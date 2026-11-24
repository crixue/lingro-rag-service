import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, model_validator
from tqdm import tqdm
from typing_extensions import Self

from configs.model_config import LOG_FORMAT, MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR, \
    INDEX_ERROR_MESSAGE_DIR_NAME
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.common_constant import EXTRACTED_TRIPLES_KEY, \
    EXTRACTED_ENTITIES_KEY, PASSAGE_KEY, PASSAGE_ID_KEY
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import extract_json_dict
from core.model_io.prompts.hipproRAG_prompts import NerPrompt
from utils.neo4j_client import Neo4jGraphDBClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


def _save_failed_traceback_to_file(error_message_filename: str, error_message_info: dict):
    error_message_filepath = os.path.join(MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR,
                                          INDEX_ERROR_MESSAGE_DIR_NAME,
                                          error_message_filename)
    error_message_dir = os.path.dirname(error_message_filepath)
    if not os.path.exists(os.path.dirname(error_message_filepath)):
        os.makedirs(error_message_dir)
    with open(error_message_filepath, 'a') as f:
        f.write(json.dumps(error_message_info) + '\n')


def _openie_rdf_recognition_and_extract(passage: str, chat_llm_client: BaseChatModel, is_chinese_text=True, id: str = None):
    """
    Extract named entities and triples from a passage using OpenAI's ChatGPT model.
    Detail refer to prompt_definitions/builtin/lang/cn/openie_post_ner_prompt.json
    """
    ner_messages = (NerPrompt.get_openie_post_ner_prompts(is_chinese_text)
                       .format_prompt(passage=passage))

    not_done = True
    retry_times = 0

    total_tokens = -1
    response_content = '{}'

    while not_done and retry_times < 2:
        try:
            retry_times += 1
            # JSON mode
            chat_completion = chat_llm_client.invoke(ner_messages.to_messages(), temperature=0,
                                                     response_format={"type": "json_object"})
            total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']

            response_content = chat_completion.content.strip().replace('\n', '')
            try:
                response_content = eval(response_content)
            except Exception as e:
                response_content = extract_json_dict(response_content)
            # else:  # no JSON mode
            #     logger.debug('[_openie_rdf_recognition_and_extract]-request llm to recognition named_entity.')
            #     chat_completion = chat_llm_client.invoke(ner_messages.to_messages(), temperature=0)
            #     total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']
            #
            #     response_content = chat_completion.content
            #     response_content = extract_json_dict(response_content)
            if 'named_entities' not in response_content or 'triples' not in response_content:
                response_content = {'named_entities': [], 'triples': []}
                logger.warning(
                    f'[_openie_rdf_recognition_and_extract] No named_entities or triples found in response_content:{response_content} of parse passage:{passage}')
                logger.warning("[_openie_rdf_recognition_and_extract]-retrying named_entity_recognition.")
            else:
                named_entities = response_content['named_entities']
                triples = response_content['triples']
                if not isinstance(named_entities, list) or not isinstance(triples, list):
                    response_content = {'named_entities': [], 'triples': []}
                    logger.warning(
                        f'[_openie_rdf_recognition_and_extract] named_entities or triples is not a list in response_content:{response_content} of parse passage:{passage}')
                logger.debug(
                    f'[_openie_rdf_recognition_and_extract] named_entities and triples found in response_content:{response_content} after parse passage:{passage}')
                not_done = False
        except Exception as e:
            logger.exception(
                f'[_openie_rdf_recognition_and_extract] Passage:{passage} NER response_content:{str(response_content)} exception throw ex:{e}')
            _save_failed_traceback_to_file(f"{id}_ner_error_messages.jsonl", {
                "passage": passage,
                "error": str(e),
                "response_content": response_content
            })

    def extract_triples(json_data):
        triples = json_data.get("triples", [])
        extracted_triples = []
        for triple in triples:
            if "subject" in triple and "predicate" in triple and "object" in triple:
                extracted_triples.append((triple["subject"], triple["predicate"], triple["object"]))
        return extracted_triples
    
    return response_content['named_entities'], extract_triples(response_content), total_tokens


def extract_openie_from_triples(documents: List[object],
                                chat_llm_client: BaseChatModel,
                                is_chinese_text: bool,
                                id: str = None,
                                ) -> tuple:
    new_json = []
    all_entities = []

    chatgpt_total_tokens = 0

    for i, item in tqdm(documents, total=len(documents), desc='Extracting OpenIE from triples'):
        passage = item[PASSAGE_KEY]
        doc_entities, triples, total_ner_tokens = _openie_rdf_recognition_and_extract(passage, chat_llm_client,
                                                                             is_chinese_text=is_chinese_text, id=id)

        doc_entities = list(np.unique(doc_entities))
        if len(doc_entities) == 0:
            logger.warning(
                f'[extract_openie_from_triples] From id:{item[PASSAGE_ID_KEY]} No entities found in doc_entities:{doc_entities}  passage:{passage} and will skip it.')
            continue
        all_entities.extend(doc_entities)
        chatgpt_total_tokens += total_ner_tokens

        item[EXTRACTED_ENTITIES_KEY] = doc_entities
        item[EXTRACTED_TRIPLES_KEY] = triples
        item.pop(PASSAGE_KEY)
        new_json.append(item)
    return (new_json, all_entities, chatgpt_total_tokens)


def remove_nodes_rels_from_graphdb_by_doc_ids(doc_ids: List[str], neo4j_client: Neo4jGraphDBClient = None,
                                              graph_db_name: str = None):
    total_deleted_nodes = 0
    total_deleted_relations = 0
    for passage_id in doc_ids:
        with neo4j_client.graph_driver.session(database=graph_db_name) as session:
            query_delete_single_passage_relations = """
                    match (e1:Entity)-[r]->(e2:Entity)
                    where apoc.coll.toSet([$passageId]) = e1.passageId or apoc.coll.toSet([$passageId]) = e2.passageId
                    delete r
                    return count(r) as deleted_relation_count
                """
            result = session.run(query_delete_single_passage_relations, parameters={'passageId': passage_id})
            for record in result:
                total_deleted_relations += record["deleted_relation_count"]

            query_delete_single_passage_nodes = """
                    match (e1:Entity)
                    where apoc.coll.toSet([$passageId]) = e1.passageId
                    delete e1
                    return count(e1) as deleted_node_count
                """
            result1 = session.run(query_delete_single_passage_nodes, parameters={'passageId': passage_id})
            for record in result1:
                total_deleted_nodes += record["deleted_node_count"]

            query_remove_passage_id_in_properties = """
                    MATCH (e:Entity)-[r]->(e2:Entity)
                    WHERE $passageId IN e.passageId
                       OR $passageId IN e2.passageId
                       OR $passageId IN r.passageId
                    FOREACH (ignore IN CASE WHEN $passageId IN e.passageId THEN [1] ELSE [] END |
                        SET e.passageId = apoc.coll.removeAll(e.passageId, [$passageId]))
                    FOREACH (ignore IN CASE WHEN $passageId IN e2.passageId THEN [1] ELSE [] END |
                        SET e2.passageId = apoc.coll.removeAll(e2.passageId, [$passageId]))
                    FOREACH (ignore IN CASE WHEN $passageId IN r.passageId THEN [1] ELSE [] END |
                        SET r.passageId = apoc.coll.removeAll(r.passageId, [$passageId]))
                """
            session.run(query_remove_passage_id_in_properties, parameters={'passageId': passage_id})

    logger.info(
        f'[remove_nodes_rels_from_graphdb_by_doc_ids] Total deleted nodes:{total_deleted_nodes} and relations:{total_deleted_relations}')


class OpenIEWithRetrievalOption(BaseModel):
    chat_llm_client: BaseChatModel = None
    is_chinese_text: bool = True
    dataset: str = None
    llm_model_name: str = 'default'
    vector_dir: str = 'data/lm_vectors'
    retriever_name: str = None
    # after init
    linking_retriever_name_processed: str = None

    @model_validator(mode='before')
    def pre_validate(cls, values):
        assert values.get('dataset') is not None, "Dataset must be provided"
        assert values.get('chat_llm_client') is not None, "Chat LLM client must be provided"
        return values

    @model_validator(mode='after')
    def after_validator(self) -> Self:
        self.linking_retriever_name_processed = self.retriever_name.replace('/', '_').replace('.', '')
        return self

    class Config:
        arbitrary_types_allowed = True

    def extract_openie_from_document(self, documents: List[Document] = None, num_processes=8, num_passages='all',
                                     is_chinese_text=True, **kwargs) -> tuple:
        retrieval_corpus = []
        for doc in documents:
            retrieval_corpus.append({PASSAGE_KEY: doc.page_content, PASSAGE_ID_KEY: doc.id})
        # for document in retrieval_corpus:
        #     document[PASSAGE_KEY] = document['title'] + '\n' + document['text']

        dataset = '_' + self.dataset

        if num_passages == 'all':
            num_passages = len(retrieval_corpus)
        else:
            try:
                num_passages = int(num_passages)
            except:
                assert False, "Set 'num_passages' to an integer or 'all'"

        flag_names = ['ner']
        flags_present = ['ner']
        if len(flags_present) > 0:
            arg_str = '_'.join(flags_present) + '_' + self.llm_model_name.replace('/', '_') + f'_{num_passages}'
        else:
            arg_str = self.llm_model_name.replace('/', '_') + f'_{num_passages}'

        # client = init_langchain_model(args.llm, model_name)  # LangChain model
        already_done = False

        # Loading files which would reduce API consumption
        extracted_triples_subset = retrieval_corpus[:num_passages]

        num_processes = min(num_processes, len(extracted_triples_subset))
        splits = np.array_split(range(len(extracted_triples_subset)), num_processes)

        args = []

        for split in splits:
            args.append(([(i, extracted_triples_subset[i],) for i in split], self.chat_llm_client, is_chinese_text,
                         kwargs.get('id', None)))

        if num_processes > 1:
            import contextvars
            ctx = contextvars.copy_context()

            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(lambda arg=arg: ctx.run(lambda: extract_openie_from_triples(*arg))) for
                    arg in args]
                outputs = [future.result() for future in as_completed(futures)]
        else:
            outputs = [extract_openie_from_triples(*arg) for arg in args]

        all_entities = []
        lm_total_tokens = 0

        for output in outputs:
            all_entities.extend(output[1])
            lm_total_tokens += output[2]

        logger.info(
            f'[extract_openie_from_document] Total entities num:{len(all_entities)} cost tokens:{lm_total_tokens}')
        return outputs, lm_total_tokens

    def create_neo4j_graph_db(self, neo4j_client: Neo4jGraphDBClient = None, graph_db_name: str = None):
        with neo4j_client.graph_driver.session() as session:
            session.run(f"CREATE DATABASE {graph_db_name} IF NOT EXISTS")

    def save_extract_openie_2_neo4j_graph(self, extracted_openie_list: List, num_passages='all',
                                          neo4j_client: Neo4jGraphDBClient = None, graph_db_name: str = None
                                          ) -> tuple:
        new_json = []
        graph = {}  # (Num Phrases, Num Phrases)

        unique_phrases = set()
        with neo4j_client.graph_driver.session(database=graph_db_name) as session:
            for triple_list in tqdm(extracted_openie_list, total=len(extracted_openie_list), desc='Save extracting triples to Neo4j'):
                if triple_list is None or len(triple_list) == 0 or len(triple_list[0]) is None:
                    continue

                for item in triple_list[0]:
                    passage_id = item[PASSAGE_ID_KEY]
                    extracted_triples = item.get(EXTRACTED_TRIPLES_KEY, None)
                    if passage_id is None or extracted_triples is None:
                        logger.warning(
                            f'[save_extract_openie_2_neo4j_graph] Error in extracting triples from item:{item}, Because passage_id or extracted_triples is None.')
                        continue

                    extracted_triples_id_list = []
                    extracted_entities_id_list = set()
                    for extracted_triple in extracted_triples:
                        if extracted_triple and len(extracted_triple) >= 3:
                            head = str(extracted_triple[0]).lower()
                            relation = str(extracted_triple[1]).lower()
                            tail = str(extracted_triple[-1]).lower()
                            query = """
                                MERGE (e1:Entity {name: $head})
                                MERGE (e2:Entity {name: $tail})
                                MERGE (e1)-[r:""" + 'rel_' + re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5_]', '_', str(relation)) + """ {name: $relation}]->(e2)
                                ON CREATE SET r.weight = 1.0,
                                    e1.dataSource = apoc.coll.toSet([$dataset]), e2.dataSource = apoc.coll.toSet([$dataset]),
                                    e1.passageId = apoc.coll.toSet($passageId), e2.passageId = apoc.coll.toSet($passageId), r.passageId = $passageId,
                                    e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                                ON MATCH SET r.weight = r.weight + 1.0,
                                    e1.passageId = apoc.coll.toSet(e1.passageId + $passageId), 
                                    e2.passageId = apoc.coll.toSet(e2.passageId + $passageId),
                                    r.passageId = apoc.coll.toSet(r.passageId + $passageId),
                                    e1.dataSource = apoc.coll.toSet(e1.dataSource + $dataset),
                                    e2.dataSource = apoc.coll.toSet(e2.dataSource + $dataset),
                                    r.dataSource = apoc.coll.toSet(r.dataSource + $dataset),
                                    e1.updatedAt = datetime(), e2.updatedAt = datetime(), r.updatedAt = datetime()
                                RETURN e1, r, e2
                            """
                            result = session.run(query, parameters={'head': head, 'relation': relation, 'tail': tail,
                                                                    'passageId': [passage_id], 'dataset': self.dataset})
                            unique_phrases.add(head)
                            unique_phrases.add(tail)
                            for record in result:
                                extracted_triples_id_list.append((record['e1'].element_id,
                                                                  'rel_' + str(record['r'].element_id),
                                                                  record['e2'].element_id))
                                extracted_entities_id_list.add(record['e1'].element_id)
                                extracted_entities_id_list.add(record['e2'].element_id)
                        else:
                            logger.warning(
                                f'[save_extract_openie_2_neo4j_graph] Error in extracting triples from item:{str(extracted_triple)}, skip it.')
                        new_json.append({
                            PASSAGE_ID_KEY: passage_id,
                            EXTRACTED_TRIPLES_KEY: extracted_triples_id_list,
                            EXTRACTED_ENTITIES_KEY: list(extracted_entities_id_list)
                        })

        # Current Cost
        # extra_info_json = {"docs": new_json,
        #                    "num_tokens": lm_total_tokens,
        #                    }
        # flags_present = ['ner']
        # if len(flags_present) > 0:
        #     arg_str = '_'.join(flags_present) + '_' + self.llm_model_name.replace('/', '_') + f'_{num_passages}'
        # else:
        #     arg_str = self.llm_model_name.replace('/', '_') + f'_{num_passages}'
        # output_path = '{}/openie{}_results_{}.json'.format(self.output_dir, self.dataset, arg_str)
        # if not Path(output_path).parent.exists():
        #     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # with open(output_path, 'w') as f:
        #     json.dump(extra_info_json, f, ensure_ascii=False)
        #     logger.info('OpenIE saved to {}'.format(output_path))

        return new_json, list(unique_phrases)
