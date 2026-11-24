import logging
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from configs.model_config import LOG_FORMAT
from core.data_connections.retrievers.knowlege_graph_retrievers.hipproRAG.processing import extract_json_dict
from core.model_io.prompts.hipproRAG_prompts import NerPrompt

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


def query_ner_named_entity_recognition(client: BaseChatModel, question: str, is_chinese_text=True):
    query_ner_prompts = NerPrompt.get_query_ner_prompts(is_chinese_text)
    query_ner_messages = query_ner_prompts.format_prompt(question=question)

    json_mode = False
    if isinstance(client, ChatOpenAI):  # JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
        response_content = chat_completion.content
        json_mode = True
    else:  # no JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'])
        response_content = chat_completion.content
        response_content = extract_json_dict(response_content)

    if not json_mode:
        try:
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except Exception as e:
            logger.exception(f'[named_entity_recognition] Query NER exception throw ex:{e}')
            response_content = {'named_entities': []}

    return response_content


def run_ner_on_texts(client, texts):
    ner_output = []

    for text in tqdm(texts):
        ner = query_ner_named_entity_recognition(client, text)
        ner_output.append(ner)

    return ner_output


def named_entity_extraction_parallel(client: BaseChatModel, dataset: str = None, queries: List[str] = [], num_processes=10):
    queries_df = pd.read_json('{}')
    output_file = 'output/{}_queries.named_entity_output.tsv'.format(dataset)
    for query in queries:
        queries_df = queries_df.append({'question': query}, ignore_index=True)

    try:
        # if dataset is not None:
        #     queries_df = pd.read_json(f'data/{dataset}.json')
        #
        #     if 'hotpotqa' in dataset:
        #         queries_df = queries_df[['question']]
        #         queries_df['0'] = queries_df['question']
        #         queries_df['query'] = queries_df['question']
        #         query_name = 'query'
        #     else:
        #         query_name = 'question'
        #
        #     try:
        #         output_df = pd.read_csv(output_file, sep='\t')
        #     except:
        #         output_df = []
        #
        #     if len(queries_df) != len(output_df):
        #         queries = queries_df[query_name].values
        #     else:
        #         logger.info('[named_entity_extraction_parallel]-Passage NER already saved to', output_file)

        num_processes = min(num_processes, len(queries))
        splits = np.array_split(range(len(queries)), num_processes)

        args = []
        for split in splits:
            args.append([queries[i] for i in split])


        if num_processes == 1:
            outputs = [run_ner_on_texts(client, args[0])]
        else:
            partial_func = partial(run_ner_on_texts, client)
            with Pool(processes=num_processes) as pool:
                outputs = pool.map(partial_func, args)

        query_triples = []

        for output in outputs:
            query_triples.extend(output)

        queries_df['triples'] = query_triples
        queries_df.to_csv(output_file, sep='\t')
        logger.info(f'[named_entity_extraction_parallel]-Passage NER saved to {output_file}' )

    except Exception as e:
        logger.exception(f'[named_entity_extraction_parallel] No queries will be processed for later retrieval and throw ex:{e}')
