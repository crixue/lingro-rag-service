import os
import pickle
from typing import List

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from configs.model_config import BM25_INDEX_ROOT_PATH, logger


class BM25InMemoryApi(BM25Okapi):
    def __init__(self, idf, doc_freqs, avgdl, corpus_size, doc_len, tokenizer=None, tokenized_docs=None, k1=1.5, b=0.75, epsilon=0.25, **kwargs):
        self.idf = idf
        self.doc_freqs = doc_freqs
        self.avgdl = avgdl
        self.corpus_size = corpus_size
        self.doc_len = doc_len
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.kwargs = kwargs
        self.tokenized_docs = tokenized_docs


class BM25DocumentIndex:
    def __init__(self, saved_folder_name: str,):
        self.bm25_index_root_path = BM25_INDEX_ROOT_PATH
        self.saved_folder_name = saved_folder_name
        self.index_to_doc: dict[str, Document] = {}  # key: index num; value: doc_id;
        self.bm25_index = None

    def _get_latest_version(self, saved_folder_name: str) -> int:
        fp = os.path.join(self.bm25_index_root_path, saved_folder_name)
        if not os.path.exists(fp):
            return 0
        versions = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(fp) if f.startswith('bm25_index_')]
        return max(versions) if versions else 0

    def _save_index(self, bm25_index, tokenized_docs: list, folder_name: str, **kwargs):
        # 保存核心参数
        bm25_params = {
            "idf": bm25_index.idf,
            "doc_freqs": bm25_index.doc_freqs,
            "avgdl": bm25_index.avgdl,
            "corpus_size": bm25_index.corpus_size,
            "doc_len": bm25_index.doc_len,
            "tokenized_docs": tokenized_docs,  # 可选，保存原始分词结果用于后续更新
            "index_to_doc": kwargs.get("index_to_doc", self.index_to_doc),
            **kwargs
        }

        latest_version = self._get_latest_version(folder_name)
        new_version = latest_version + 1
        saved_file_name = f'bm25_index_{new_version}.pkl'
        fp = os.path.join(self.bm25_index_root_path, folder_name, saved_file_name)
        if not os.path.exists(fp):
            os.makedirs(os.path.join(self.bm25_index_root_path, folder_name), exist_ok=True)

        with open(fp, "wb") as f:
            pickle.dump(bm25_params, f)
            logger.info(f"[_save_index]-BM25 index saved to {fp}")
        # 删除旧版本，只保留当前版本和上一个版本
        fp_parent_dir = os.path.join(self.bm25_index_root_path, folder_name)
        versions = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(fp_parent_dir) if f.startswith('bm25_index_')]
        versions.sort(reverse=True)
        for version in versions[2:]:
            removed_file = os.path.join(self.bm25_index_root_path, folder_name, f'bm25_index_{version}.pkl')
            os.remove(removed_file)
            logger.info(f"[_save_index]-BM25 index removed {removed_file}")

    def _if_exists_index(self) -> bool:
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"
        fp = os.path.join(self.bm25_index_root_path, self.saved_folder_name)
        if not os.path.exists(fp):
            return False

        latest_version = self._get_latest_version(self.saved_folder_name)
        saved_file_name = f'bm25_index_{latest_version}.pkl'
        fp = os.path.join(self.bm25_index_root_path, self.saved_folder_name, saved_file_name)
        if os.path.exists(fp):
            return True
        return False

    def load_index(self) -> BM25Okapi:
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"
        latest_version = self._get_latest_version(self.saved_folder_name)
        saved_file_name = f'bm25_index_{latest_version}.pkl'
        fp = os.path.join(self.bm25_index_root_path, self.saved_folder_name, saved_file_name)
        if not os.path.exists(fp):
            logger.warning(f"[load_index]-BM25 index folder {fp} not exists and will be loaded.")
            return None

        with open(fp, "rb") as f:
            bm25_params = pickle.load(f)
        bm25_index = BM25InMemoryApi(**bm25_params)
        self.bm25_index = bm25_index
        self.index_to_doc = bm25_params.get("index_to_doc", {})
        logger.info(f"[load_index]-BM25 index loaded from {fp}")
        return bm25_index

    def build_index(self, is_chinese_text: bool, documents: List[Document], ) -> BM25Okapi:
        """
        Build the BM25 index from the documents.
        """
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"
        tokenized_docs = []
        for index, doc in enumerate(documents):
            if doc.page_content:
                if is_chinese_text:
                    tokenized_doc = list(jieba.cut(doc.page_content))
                else:
                    tokenized_doc = doc.page_content.split()
                tokenized_docs.append(tokenized_doc)
                doc.page_content = ""
                self.index_to_doc[str(index)] = doc
        bm25 = BM25Okapi(tokenized_docs)
        #save the index
        self._save_index(bm25, tokenized_docs, self.saved_folder_name, is_chinese_text=is_chinese_text)
        self.bm25_index = bm25
        return bm25

    def full_update_index(self, documents: List[Document]) -> BM25Okapi:
        """
        Full update the BM25 index from the documents.
        """
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"

        # load original index info
        fp = os.path.join(self.bm25_index_root_path, self.saved_folder_name)
        if not os.path.exists(fp):
            logger.warning(f"[full_update_index]-BM25 index folder {fp} not exists and will be loaded.")
        with open(os.path.join(fp, f'bm25_index_{self._get_latest_version(self.saved_folder_name)}.pkl'), "rb") as f:
            bm25_params = pickle.load(f)

        is_chinese_text = bm25_params.get("is_chinese_text", False)
        # if is_chinese_text:
        #     tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        # else:
        #     tokenized_docs = [doc.split() for doc in documents]
        # tokenized_docs += bm25_params.get("tokenized_docs", [])
        tokenized_docs = bm25_params.get("tokenized_docs", [])
        original_index_to_doc = bm25_params.get("index_to_doc", {})
        original_index_count = len(bm25_params.get("index_to_doc", {}).keys())
        for index, doc in enumerate(documents):
            if doc.page_content:
                if is_chinese_text:
                    tokenized_doc = list(jieba.cut(doc.page_content))
                else:
                    tokenized_doc = doc.page_content.split()
                tokenized_docs.append(tokenized_doc)
                original_index_to_doc[str(original_index_count + index)] = doc.id

        bm25 = BM25Okapi(tokenized_docs)
        self.index_to_doc = original_index_to_doc
        #save the index
        self._save_index(bm25, tokenized_docs, self.saved_folder_name)
        self.bm25_index = bm25
        return bm25

    def create_or_update_index(self, is_chinese_text: bool, documents: List[Document]) -> BM25Okapi:
        """
        Create or update the BM25 index from the documents.
        """
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"
        if self._if_exists_index() is None:
            return self.full_update_index(documents)
        else:
            return self.build_index(is_chinese_text, documents)

    def delete_index(self):
        """
        Delete the BM25 index.
        """
        assert self.saved_folder_name is not None, "saved_folder_name should not be None"
        fp = os.path.join(self.bm25_index_root_path, self.saved_folder_name)
        if not os.path.exists(fp):
            logger.warning(f"[delete_index]-BM25 index folder {fp} not exists and will be deleted.")
        else:
            for f in os.listdir(fp):
                os.remove(os.path.join(fp, f))
            os.rmdir(fp)
            logger.info(f"[delete_index]-BM25 index folder {fp} have been deleted.")
        self.bm25_index = None

    def similarity_search_with_score(self, query: str, is_chinese_text: bool, k: int | None = None) -> list[tuple[Document, float]]:
        """
        Search the BM25 index with the query.
        """
        if self.bm25_index is None:
            self.load_index()
        if self.bm25_index is None:
            logger.error("BM25 index is None and can not be loaded.")
            return []
        if is_chinese_text:
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split()
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        doc_scores = [(i, score) for i, score in enumerate(doc_scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        doc_scores = doc_scores[:k]
        results = []
        for index, score in doc_scores:
            results.append((self.index_to_doc.get(str(index), None), score))
        return results