import unittest

from langchain_core.language_models import BaseChatModel

from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM
from utils.common_util import count_tokens


class TestDefaultLangchainLLM(unittest.IsolatedAsyncioTestCase):
    def test0(self):
        model_config = {
                "temperature": 0.5,
                "top_p": 0.9,
                "streaming": True,
                "max_tokens": 2000
            }
        langchain_llm = DefaultLangchainLLM(
            model_name="glm-4-flash-250414",
            user_id="7af04ac6",
            product_id="MODEL-0205",
            model_config=model_config)
        llm_client = langchain_llm.get_llm_client()
        response = llm_client.invoke("你好呀")
        print(response)

    def test_with_stream(self):
        model_config = {
                "temperature": 0.5,
                "top_p": 0.9,
                "streaming": True,
                "max_tokens": 2000
            }
        langchain_llm = DefaultLangchainLLM(
            model_name="glm-4-flash-250414",
            user_id="92040b49",
            product_id="MODEL-0205",
            model_config=model_config)
        llm_client: BaseChatModel = langchain_llm.get_llm_client()
        for message in llm_client.stream("你好呀"):
            print(message)


    async def test_with_stream_async(self):
        model_config = {
                "temperature": 0.5,
                "top_p": 0.9,
                "streaming": True,
                "max_tokens": 2000
            }
        langchain_llm = DefaultLangchainLLM(
            model_name="glm-4-flash-250414",
            user_id="7af04ac6",
            product_id="MODEL-0205",
            model_config=model_config)
        llm_client: BaseChatModel = langchain_llm.get_llm_client()
        async for message in llm_client.astream("你好呀"):
            print(message)

    def test_count_tokens_from_str(self):
        str_content = """
        FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets

Xiaohui Victor Li*

Imperial College London

London, United Kingdom

xiaohui.li21@alumni.imperial.ac.uk

ABSTRACT

Dynamic knowledge graphs (DKGs) are popular structures to express different types of connections between objects over time. They can also serve as an efficient mathematical tool to represent information extracted from complex unstructured data sources, such as text or images. Within financial applications, DKGs could be used to detect trends for strategic thematic investing, based on information obtained from financial news articles. In this work, we explore the properties of large language models (LLMs) as dynamic knowledge graph generators, proposing a novel open-source fine-tuned LLM for this purpose, called the Integrated Contextual Knowledge Graph Generator (ICKG). We use ICKG to produce a novel open-source DKG from a corpus of financial news articles, called FinDKG, and we propose an attention-based GNN architecture for analysing it, called KGTransformer. We test the performance of the proposed model on benchmark datasets and FinDKG, demonstrating superior performance on link prediction tasks. Additionally, we evaluate the performance of the KGTransformer on FinDKG for thematic investing, showing it can outperform existing thematic ETFs.

KEYWORDS

Dynamic knowledge graphs, graph attention networks, graph neural networks, graph transformers, large language models.

1 INTRODUCTION

A knowledge graph (KG) is a data structure that encodes information consisting in entities and different types of relations between them. Formally, a KG can be represented as $\mathcal{G} = {\mathcal{E}, \mathcal{R}, \mathcal{F}}$ , where $\mathcal{E}$ and $\mathcal{R}$ denote the sets of entities and relations respectively, and $\mathcal{F} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$ represents a set of facts, consisting in relations of different types between entities. The triplet $(s, r, o) \in \mathcal{F}$ is the fundamental building block of a KG, where $s \in \mathcal{E}$ represents the source entity, $r \in \mathcal{R}$ the relation, and $o \in \mathcal{E}$ the object entity. For instance, the triplet (OpenAI, Invent, ChatGPT) shows how entities and relations combine to form a fact, with OpenAI and ChatGPT as entities and Invent as the relation.

Temporal or dynamic knowledge graphs (DKGs) extend static KGs by incorporating temporal dynamics. Each fact in a DKG is associated with a timestamp $t\in \mathbb{R}_+$ , allowing the model to capture the temporal evolution of events. Therefore, events occur in quadruples $(s_i,r_i$ $i\leq j$ $i,j\in \mathbb{N}$ . Then, the DKG $\mathcal{G}_t = (\mathcal{E},\mathcal{R},\mathcal{F}_t)$ at time $t$ can be expressed via a time-varying set of facts $\mathcal{F}_t$ defined as

$$ \mathcal {F} _ {t} = \left{\left(s _ {i}, r _ {i}, o _ {i}, t _ {i}\right): s _ {i}, o _ {i} \in \mathcal {E}, r _ {i} \in \mathcal {R}, t _ {i} < t \right}. \tag {1} $$

Francesco Sanna Passino

Imperial College London

London, United Kingdom

f.sannapassino@imperial.ac.uk

The task of estimating a model for $\mathcal{G}_t$ from observed data is called dynamic knowledge graph learning. This typically involves data-driven training of graph neural networks, designed to model both the structure and the temporal dynamics of the KGs over time.

In the context of meta-entity meta-entities, the meta-entities are a set of meta-entities that are associated with the entity. In real-world applications such as finance, entities and relations can be further grouped into categories, often called meta-entities. For example, consider the relation between the entity Jeff Bezos which is of type Person, and the entity Amazon, which is of type Company. The relation between them is Founder Of, which could be considered to have the type Business action. In this work, inspired by heterogeneous graph transformers [HGT, 13] and EvoKG [25]. This results in the Knowledge Graph Transformer (KGTransformer), an attention-based graph neural network (GNN) designed to create dynamic lower-dimensional representations of entities and relations.

In addition to DKGs, Large Language Models (LLMs) have also been gaining popularity recently within the financial sector, demonstrating potential in enhancing various financial tasks through advanced natural language processing (NLP) capabilities [23]. Popular models such as BERT, the GPT series, and financial-specific variants such as FinBERT [2] and FinGPT [41] leverage LLMs to improve the state-of-the-art in tasks such as financial sentiment analysis.

The application of LLMs to dynamic knowledge graphs has been so far limited in the literature. Therefore, one of the main contributions of this work is to also propose a pipeline for generative knowledge graph construction (KGC) via Large Language Models (LLMs), resulting in the Integrated Contextual Knowledge Graph Generator (ICKG) large language model. In particular, we develop a fine-tuned LLM to systematically extract entities and relationships from textual data via engineered input queries or "prompts", subsequently assembling them into event quadruples of the same form as (1). We use the proposed ICKG LLM to generate an open-sourced financial knowledge graph dataset, called FinDKG.

In summary, our contributions in this work are threefold:

(1) We propose KGTransformer, an attention-based GNN architecture for dynamic knowledge graph learning that includes information about meta-entities (cf. Section 4), combining existing work on GATs [34], HGTs [13] and EvoKG [25]. We demonstrate substantial improvements in link prediction metrics (cf. Section 5.1) on real-world DKGs. (2) We develop an open-source LLM for dynamic knowledge graph generation for finance called Integrated Contextual Knowledge Graph Generator (ICKG, cf. Section 3). (3) We utilise ICKG to create an open-source dynamic knowledge graph based on financial news articles, called FinDKG

(cf. Section 3.1). FinDKG is used for thematic investing upon capitalizing on the AI trend, improving upon other AI-themed portfolios (cf. Section 5.3).

The remainder of this work is organised as follows: Section 2 discusses related literature. Next, Section 3 and 4 discuss the main contributions of our work: ICKG and KGTransformer. Finally, Section 5 discusses applications on real-world DKGs.
        """
        token_count = count_tokens(str_content)
        print(f"字符串内容的令牌数量：{token_count}")  # 输出令牌数量