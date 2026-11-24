import unittest

from core.data_connections.document_loaders.file_loader_v2 import JSONUnstructuredFileLoaderV2


class TestRetriever(unittest.TestCase):

    def test_preview_loaded_docs(self):
        """
        Test preview_loaded_docs
        """
        file_path = 'data/hotpotqa_train.json'
        file_loader = JSONUnstructuredFileLoaderV2(
            max_characters=1000,
            file_path=file_path,
            jq_schema='.',
            json_lines=False,
            text_content=False,
        )
        docs = file_loader.load()
        for doc in docs:
            print(doc.metadata)
            print(doc.page_content)
            print('---')
