from typing import List, Dict

from langchain_core.documents import Document

from configs.model_config import DEFAULT_REPLACE_WORDS_DICT
from utils.common_util import get_beijing_current_time


def inject_label_to_metadata(docs: List[Document], label_dict: dict):
    for doc in docs:
        if doc is None:
            continue
        if doc.metadata is None:
            doc.metadata = {}
        if label_dict is not None:
            for k, v in label_dict.items():
                doc.metadata.update({k: v})
    return docs

def add_created_date_on_doc_metadata(docs: List[Document]):
    for doc in docs:
        if doc is None:
            continue
        if doc.metadata is None:
            doc.metadata = {}
        current_time_str = get_beijing_current_time().strftime("%Y-%m-%d %H:%M:%S")
        doc.metadata.update({"created_at": current_time_str})
    return docs


def replace_words_on_doc_page_content(docs: List[Document], replace_words: Dict[str, str] = DEFAULT_REPLACE_WORDS_DICT):
    for doc in docs:
        if doc is None:
            continue
        if doc.page_content is None:
            continue
        for k, v in replace_words.items():
            doc.page_content = doc.page_content.replace(k, v)
    return docs