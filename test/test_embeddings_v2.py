import unittest

from configs.config_local import EMBEDDING_MODELS
from core.embeddings.default_embeddings_v2 import DefaultEmbeddingsV2, OpenAICompatibleEmbeddings
from langchain_community.vectorstores import FAISS  # 新增


class MyTestCase(unittest.TestCase):
    def test_get_default_static_embeddings(self):
        embedding_model_id = "EMBEDDING-002"
        text = "OpenAI's embedding models are powerful tools for NLP tasks."

        # 获取 embedding
        embedding, model_name = DefaultEmbeddingsV2.get_static_embeddings(
            embedding_model_id=embedding_model_id
        )
        result = embedding.embed_query(text)

        # 输出结果
        print(f"Model: {model_name}")
        print(f"Embedding: {result}")

        # 验证结果非空
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_get_static_embeddings(self):
        text = "OpenAI's embedding models are powerful tools for NLP tasks."

        # 获取 embedding
        embedding, model_name = DefaultEmbeddingsV2.get_static_embeddings(
        )
        result = embedding.embed_query(text)

        # 输出结果
        print(f"Model: {model_name}")
        print(f"Embedding: {result}")

        # 验证结果非空
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_get_static_embeddings3(self):
        embedding_model_id = "EMBEDDING-003"
        text = "OpenAI's embedding models are powerful tools for NLP tasks."

        # 获取 embedding
        embedding, model_name = DefaultEmbeddingsV2.get_static_embeddings(
            embedding_model_id=embedding_model_id
        )
        result = embedding.embed_query(text)

        # 输出结果
        print(f"Model: {model_name}")
        print(f"Embedding: {result}")

        # 验证结果非空
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_faiss_similarity_en_basic(self):
        # English similarity search
        embedding, _ = DefaultEmbeddingsV2.get_static_embeddings()
        texts = [
            "The capital of France is Paris.",
            "Machine learning enables computers to learn from data.",
            "OpenAI creates AI models."
        ]
        db = FAISS.from_texts(texts=texts, embedding=embedding)
        query = "Paris is the capital city of France."

        docs_scores = db.similarity_search_with_score(query, k=2)
        self.assertGreaterEqual(len(docs_scores), 1)
        top_doc, top_score = docs_scores[0]
        self.assertEqual(top_doc.page_content, texts[0])
        if len(docs_scores) > 1:
            self.assertLessEqual(top_score, docs_scores[1][1])

    def test_faiss_similarity_zh_basic(self):
        embedding_config = EMBEDDING_MODELS[1]
        # 中文相似度检索
        embedding = OpenAICompatibleEmbeddings(
            user_id ="92040b49",
            openai_api_base = embedding_config.get("api_base"),
            model=embedding_config.get("model_name"),
            embedding_model_id=embedding_config.get("id"),
            api_key=embedding_config.get("api_key"),
        )
        texts = [
            "北京是中国的首都。",
            "苹果是一家科技公司。",
            "机器学习是一种让计算机从数据中学习的方法。"
        ]
        db = FAISS.from_texts(texts=texts, embedding=embedding)
        query = "中国的首都是北京。"

        docs_scores = db.similarity_search_with_score(query, k=2)
        for doc, score in docs_scores:
            print(f"Doc: {doc.page_content}, Score: {score}")
        self.assertGreaterEqual(len(docs_scores), 1)
        top_doc, top_score = docs_scores[0]
        self.assertEqual(top_doc.page_content, texts[0])
        if len(docs_scores) > 1:
            self.assertLessEqual(top_score, docs_scores[1][1])

    def test_faiss_similarity_en_ml(self):
        # English: ML-related query should match ML sentence
        embedding_config = EMBEDDING_MODELS[0]
        # 中文相似度检索
        embedding = OpenAICompatibleEmbeddings(
            user_id ="92040b49",
            openai_api_base = embedding_config.get("api_base"),
            model=embedding_config.get("model_name"),
            embedding_model_id=embedding_config.get("id"),
            api_key=embedding_config.get("api_key"),
        )
        texts = [
            "OpenAI builds powerful AI systems.",
            "Machine learning enables computers to learn from data.",
            "Paris is a beautiful city in France."
        ]
        db = FAISS.from_texts(texts=texts, embedding=embedding)
        query = "Algorithms that learn from data improve computer performance."

        docs_scores = db.similarity_search_with_score(query, k=2)
        for doc, score in docs_scores:
            print(f"Doc: {doc.page_content}, Score: {score}")
        self.assertGreaterEqual(len(docs_scores), 1)
        top_doc, top_score = docs_scores[0]
        self.assertEqual(top_doc.page_content, texts[1])
        if len(docs_scores) > 1:
            self.assertLessEqual(top_score, docs_scores[1][1])

    def test_faiss_similarity_by_vector_en(self):
        # 基于向量的英文相似度检索（与 search_faiss 一致：先 embed_query，再 by_vector 检索）
        embedding, _ = DefaultEmbeddingsV2.get_static_embeddings()
        texts = [
            "The capital of France is Paris.",
            "Machine learning enables computers to learn from data.",
            "OpenAI creates AI models."
        ]
        db = FAISS.from_texts(texts=texts, embedding=embedding)

        query = "Paris is the capital city of France."
        query_vec = embedding.embed_query(query)
        docs_scores = db.similarity_search_with_score_by_vector(query_vec, k=2)

        self.assertGreaterEqual(len(docs_scores), 1)
        top_doc, top_score = docs_scores[0]
        print(f"[EN-by-vector] top: {top_doc.page_content} | score: {top_score}")
        if len(docs_scores) > 1:
            print(f"[EN-by-vector] second: {docs_scores[1][0].page_content} | score: {docs_scores[1][1]}")
        self.assertEqual(top_doc.page_content, texts[0])
        if len(docs_scores) > 1:
            self.assertLessEqual(top_score, docs_scores[1][1])

    def test_faiss_similarity_by_vector_zh(self):
        # 基于向量的中文相似度检索（与 search_faiss 一致：先 embed_query，再 by_vector 检索）
        embedding_config = EMBEDDING_MODELS[0]
        embedding = OpenAICompatibleEmbeddings(
            user_id="92040b49",
            openai_api_base=embedding_config.get("api_base"),
            model=embedding_config.get("model_name"),
            embedding_model_id=embedding_config.get("id"),
            api_key=embedding_config.get("api_key"),
        )
        texts = [
            "北京是中国的首都。",
            "苹果是一家科技公司。",
            "机器学习是一种让计算机从数据中学习的方法。"
        ]
        db = FAISS.from_texts(texts=texts, embedding=embedding)

        query = "中国的首都是北京。"
        query_vec = embedding.embed_query(query)
        docs_scores = db.similarity_search_with_score_by_vector(query_vec, k=2)
        for doc, score in docs_scores:
            print(f"Doc: {doc.page_content}, Score: {score}")

        self.assertGreaterEqual(len(docs_scores), 1)
        top_doc, top_score = docs_scores[0]
        print(f"[ZH-by-vector] top: {top_doc.page_content} | score: {top_score}")
        if len(docs_scores) > 1:
            print(f"[ZH-by-vector] second: {docs_scores[1][0].page_content} | score: {docs_scores[1][1]}")
        self.assertEqual(top_doc.page_content, texts[0])
        if len(docs_scores) > 1:
            self.assertLessEqual(top_score, docs_scores[1][1])

