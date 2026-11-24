import json
import os
import unittest

from configs.model_config import BASE_DIR
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.data_connections.retrievers.storage.documents_storage import DefaultDocumentsStorage


class TestDocumentsSaver(unittest.TestCase):
    document_id_prefix = 'test_extract_doc'

    def test_save_documents(self):
        sample0_fp = os.path.join(BASE_DIR, "samples", "0-words-cn.txt")
        fileLoader = DefaultFileLoaderComponent(file_path=sample0_fp, cn_chunk_size=200,
                                                is_chinese_text=True, mode='single', )
        docs = fileLoader.load_and_split()
        DefaultDocumentsStorage(documents=docs,
                                root_path='./test_data/test_saver',
                                document_id_prefix=self.document_id_prefix
                                ).save_documents()

    def test_retriever_documents(self):
        retrieve_ids = [self.document_id_prefix + "_4", self.document_id_prefix + "_2"]
        docs = (DefaultDocumentsStorage(root_path='./test_data/test_extract_doc', document_id_prefix=self.document_id_prefix)
         .retriever_documents(retrieve_ids))
        for doc in docs:
            print(json.dumps(doc.model_dump(), ensure_ascii=False, indent=4))

    def test_load_full_content(self):
        """测试 load_full_content 方法加载完整文档内容"""
        sample0_fp = os.path.join(BASE_DIR, "samples", "1-many-words.docx")
        fileLoader = DefaultFileLoaderComponent(file_path=sample0_fp)

        # 加载完整内容
        doc = fileLoader.load_full_content(include_metadata=True)

        # 打印内容预览
        print(f"\n完整文档长度: {len(doc.page_content)} 字符")
        print(f"元数据: {json.dumps(doc.metadata, ensure_ascii=False, indent=2)}")
        print(f"内容预览: ...{doc.page_content[-300:]}")
