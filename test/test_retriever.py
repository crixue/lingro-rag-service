import os
import unittest
from typing import List

import numpy as np
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from mpmath.calculus.extrapolation import nprod
from rapidfuzz.distance.Prefix import similarity

from configs.model_config import BASE_DIR
from core.data_connections.document_loaders.file_loader_util import DefaultFileLoaderComponent
from core.data_connections.retrievers.default_retriever_wrapper_v2 import ElasticSearchRetrieverWrapperV2
from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import CustomLocalVectorDecorator
from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2


class TestRetriever(unittest.TestCase):

    @staticmethod
    def split_callback(docs: List[Document]):
        items: List[Document] = []
        for item in docs:
            import re
            item.page_content = re.sub(r'\s+', ' ', item.page_content)
            items.append(item)
        return items

    def test_add_documents_to_custom_faiss(self):
        is_chinese_text = True  # 是否是中文文本
        sample0_fp = os.path.join(BASE_DIR, "samples", "0-words-cn.txt")
        fileLoader = DefaultFileLoaderComponent(file_path=sample0_fp, cn_chunk_size=300,
                                                is_chinese_text=is_chinese_text, mode='single',
                                                split_callback=TestRetriever.split_callback)
        docs = fileLoader.load_and_split()
        inst = CustomLocalVectorDecorator(folder_name="test_retrievers_custom_faiss", is_chinese_text=is_chinese_text)
        faiss_instance = FAISS.from_documents(docs, inst.embeddings)
        faiss_instance.save_local(folder_path=inst.current_vs_path, )


    def test_load_custom_faiss(self):
        is_chinese_text = True  # 是否是中文文本
        instance = CustomLocalVectorDecorator(folder_name="test_retrievers_custom_faiss", is_chinese_text=is_chinese_text)
        faiss_instance = instance.local_vector_stores()
        docs = faiss_instance.similarity_search("人工智能将在亚洲经济中发挥什么样的作用？", k=2)
        print(docs[0].page_content)

    def test_custom_faiss_add_docs(self):
        # Imports
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.storage import InMemoryStore
        is_chinese_text = True  # 是否是中文文本

        sample0_fp = os.path.join(BASE_DIR, "samples", "0-words-cn.txt")
        fileLoader = DefaultFileLoaderComponent(file_path=sample0_fp, cn_chunk_size=2000,
                                                is_chinese_text=is_chinese_text, mode='single',
                                                split_callback=TestRetriever.split_callback)
        docs = fileLoader.load_and_split()

        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        # This text splitter is used to create the child documents
        # It should create documents smaller than the parent
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        vs_folder_name = "test_retrievers_custom_faiss"  # 矢量数据库文件夹名称, 如果不想将各个不同的文件加载到同一个矢量数据库中，可以设置不同的文件夹名称
        instance = CustomLocalVectorDecorator(folder_name="test_retrievers_custom_faiss", is_chinese_text=is_chinese_text)
        vectorstore = instance.local_vector_stores()
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(documents=docs, ids=None, add_to_docstore=True)
        sub_docs = vectorstore.similarity_search("人工智能将在亚洲经济中发挥什么样的作用？", k=3)
        for doc in sub_docs:
            print(doc.page_content)

        print("=====================================")
        all_docs = retriever.get_relevant_documents("人工智能将在亚洲经济中发挥什么样的作用？", k=3)
        for doc in all_docs:
            print(doc.page_content)

    def test_read_all_parent_docs(self):
        parent_docs_dir = r"./test_data/vector_store/text2vec/7VpWqBTc/ppt-test2_doc_store"
        file_names = os.listdir(parent_docs_dir)
        for file_name in file_names:
            file_path = os.path.join(parent_docs_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f.read())


    def test_embedding_model(self):
        embedding_config = DefaultEmbeddingsV2.get_default_embedding_config()
        embedding_model = OpenAIEmbeddings(openai_api_base="http://localhost:48077/v1",
                                     model="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
                                     api_key=embedding_config['api_key'],
                                     max_retries=embedding_config.get('max_retries', None),
                                     dimensions=embedding_config.get('dimensions', None),
                                     )
        doc_snippet1 = """
        示例半导体在2024年实现了28纳米工艺的量产，良率稳定在九成以上。公司表示，该节点主要用于电源管理与显示驱动芯片，客户导入周期约为两个季度。
        """
        doc_snippet2 = """
        受出口管制与供应链调整影响，多家晶圆厂重新评估了成熟制程的产能布局。示例半导体同期宣布扩产，并强调设备与材料的本地化比例正在提升。
        """
        doc_snippet3 = """
        本周多地将出现明显降温，农业部门提醒种植户关注夜间低温对幼苗的影响，并建议提前做好大棚保温措施。
        """
        doc_snippet4 = """
        示例半导体宣布28纳米工艺实现量产。
        """
        query = "示例半导体的28纳米工艺量产了吗"
        embedding = embedding_model.embed_query(query)
        doc_snippet_embedding = embedding_model.embed_query(doc_snippet1)
        similarity_score1 = np.dot(embedding, doc_snippet_embedding)
        print(similarity_score1)

        similarity_score2 = np.dot(embedding, embedding_model.embed_query(doc_snippet2))
        print(similarity_score2)

        similarity_score3 = np.dot(embedding, embedding_model.embed_query(doc_snippet3))
        print(similarity_score3)

        similarity_score4 = np.dot(embedding, embedding_model.embed_query(doc_snippet4))
        print(similarity_score4)

    def test_load_custom_faiss1(self):
        is_chinese_text = False  # 是否是中文文本
        instance = CustomLocalVectorDecorator(folder_name="e3bb0285/test-local-vs-index-v3",
                                              is_chinese_text=is_chinese_text,
                                              )
        query = "Which singer is American, Mark King or Nick Hexum?"
        # faiss_instance = instance.local_vector_stores()
        # docs = faiss_instance.similarity_search(query, k=4)
        docs = instance.as_retriever(is_chinese_text=is_chinese_text).get_relevant_documents(query)
        # docs = faiss_instance.max_marginal_relevance_search(query, k=4)
        for doc in docs:
            print(doc)

    def test_elasticsearch_retriever(self):
        user_folder_id = "f4c7269a"
        query = "Grace Krilanovich debut novel name publisher city"
        retriever = ElasticSearchRetrieverWrapperV2(user_folder_id=user_folder_id, top_k=2).get_retriever()
        docs = retriever.get_relevant_documents(query)
        for doc in docs:
            print(doc)