import os
import pickle
import unittest

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from configs.model_config import BASE_DIR, CHARACTER_TEXT_SPLITTER
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from utils.common_util import count_tokens

try:
    # 优先从现有工具模块导入，如路径不同请自行调整
    from core.data_connections.document_loaders.file_loader_util import DefaultUnstructuredFileLoaderV2
except Exception:
    DefaultUnstructuredFileLoaderV2 = None


class TestDocumentLoader(unittest.TestCase):

    # -------------------- DefaultUnstructuredFileLoaderV2 Tests --------------------
    def _run_v2_case(self, file_path: str, max_characters: int, mode: str, chunking_strategy: str):
        if DefaultUnstructuredFileLoaderV2 is None:
            self.skipTest("DefaultUnstructuredFileLoaderV2 未找到，跳过。")
        if not file_path or not os.path.exists(file_path):
            self.skipTest("请手动设置有效的 file_path 后再运行此测试。")
        preferred_max_characters = max(1, int(max_characters * 0.6))

        loader = DefaultUnstructuredFileLoaderV2(
            file_path=file_path,
            mode=mode,
            chunking_strategy=chunking_strategy,
            preferred_max_characters=preferred_max_characters,
            max_characters=max_characters,
            # 需要可加：include_orig_elements=False, languages=["ch", "en"], strategy="auto"
        )
        docs = loader.load()
        self.assertIsInstance(docs, list)
        self.assertTrue(len(docs) >= 1)
        self.assertIsInstance(docs[0], Document)
        for doc in docs:
            print(doc.model_dump())

    def test_v2_single_basic(self):
        file_path = os.path.join(BASE_DIR, "samples", "word_with_dic_tables.docx")
        max_characters = CHARACTER_TEXT_SPLITTER.get("cn_chunk_size", 3000)
        self._run_v2_case(file_path, max_characters, mode="single", chunking_strategy="basic")

    def test_v2_single_by_title(self):
        file_path = os.path.join(BASE_DIR, "samples", "word_with_dic_tables.docx")
        max_characters = CHARACTER_TEXT_SPLITTER.get("cn_chunk_size", 3000)
        self._run_v2_case(file_path, max_characters, mode="single", chunking_strategy="by_title")

    def test_v2_elements_basic(self):
        file_path = os.path.join(BASE_DIR, "samples", "word_with_dic_tables.docx")
        max_characters = CHARACTER_TEXT_SPLITTER.get("cn_chunk_size", 3000)
        self._run_v2_case(file_path, max_characters, mode="elements", chunking_strategy="basic")

    def test_v2_elements_by_title(self):
        original_file = os.path.join(BASE_DIR, "samples", "word_with_dic_tables.docx")
        max_characters = CHARACTER_TEXT_SPLITTER.get("en_chunk_size", 3000)
        self._run_v2_case(original_file, max_characters, mode="elements", chunking_strategy="by_title")

    # -------------------- End V2 Tests --------------------

    def test_load_txt(self):
        '''
        new_after_n_chars: int (default=max_characters) - the “soft” maximum size for a chunk. A chunk that already exceeds this number of characters will not be extended, even if the next element would fit without exceeding the specified hard maximum. This can be used in conjunction with max_characters to set a “preferred” size, like “I prefer chunks of around 1000 characters, but I’d rather have a chunk of 1500 (max_characters) than resort to text-splitting”. This would be specified with (..., max_characters=1500, new_after_n_chars=1000).
        '''
        load_file_path = os.path.join(BASE_DIR, "samples", "2-words-cn.txt")
        loader = UnstructuredLoader(
            load_file_path,
            chunking_strategy="basic",
            new_after_n_chars=500,
            max_characters=800,
            include_orig_elements=False,
        )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_load_html(self):
        load_file_path = os.path.join(BASE_DIR, "samples", "春节_百度百科.html")
        loader = UnstructuredLoader(
            load_file_path,
            chunking_strategy="by_title",
            new_after_n_chars=200,
            max_characters=800,
            include_orig_elements=False,
        )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_load_docx_by_title(self):
        load_file_path = os.path.join(BASE_DIR, "samples", "1-many-words.docx")
        loader = UnstructuredLoader(
            load_file_path,
            chunking_strategy="by_title",
            new_after_n_chars=300,
            max_characters=500,
            include_orig_elements=False,
        )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_load_pdf_require_ocr(self):
        load_file_path = os.path.join(BASE_DIR, "samples", "0-words-with-pics.pdf")
        os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
        loader = DefaultFileLoaderComponent(file_path=load_file_path,
                                            mode="single",
                                            strategy="auto",
                                            languages=["ch", "en"],
                                            preferred_max_characters=200,
                                            max_characters=300,
                                            )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_load_image_require_ocr(self):
        os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
        load_file_path = os.path.join(BASE_DIR, "samples", "test_image_with_std_cn_words.png")
        loader = DefaultFileLoaderComponent(file_path=load_file_path,
                                            mode="single",
                                            strategy="hi_res",
                                            languages=["ch", "en"],
                                            preferred_max_characters=200,
                                            max_characters=300,
                                            )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_default_file_loader_component(self):
        load_file_path = os.path.join(BASE_DIR, "samples", "analysis_docs", "FinDKG.md")
        loader = DefaultFileLoaderComponent(
            file_path=load_file_path,
            chunking_strategy="by_title",
            preferred_max_characters=12000,
            max_characters=15000,
        )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())
            print(f"---doc tokens: {len(doc.page_content)}---")

    def test_load_jsonl(self):
        load_file_path = os.path.join(BASE_DIR, "data", "processed_data", "hotpotqa_corpus.jsonl")
        loader = DefaultFileLoaderComponent(
            file_path=load_file_path,
            jq_schema=".",
            json_lines=True,
            text_content=False,
            preferred_max_characters=300,
            max_characters=600,
        )
        docs = loader.load()
        for doc in docs:
            print(doc.model_dump())

    def test_pickle(self):
        from typing import List
        docs: List[Document] = [
            Document(
                page_content="text,"),
        ]
        msg_info = {
            "docs": docs,
        }

        with open("test.p", "wb") as f:
            pickle.dump(msg_info, f)

        with open("test.p", "rb") as f:
            msg_info = pickle.load(f)
            print(msg_info)