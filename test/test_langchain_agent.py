import json
import os
import unittest

from json_repair import repair_json
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from configs.model_config import BASE_DIR, config
from core.model_io.llm_models.default_langchain_llm import DefaultLangchainLLM


class KeyEntity(BaseModel):
    entity_name: str
    entity_type: str
    value: str
    context: str


class AbstractResponse(BaseModel):
    core_summary: str = Field(description="全文核心摘要，50-300字")
    theme_tags: list[str] = Field(description="主题标签列表")
    document_intent_type: str = Field(description="文档意图和类型")
    key_entities: list[KeyEntity] = Field(description="关键实体与数值列表")
    citation_references: dict[str, str] = Field(description="引用解析字典")
    global_glossary: dict[str, str] = Field(description="全局术语表")


class MyTestCase(unittest.TestCase):
    def setUp(self):
        model_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "streaming": False,
            "max_tokens": 2400,
            "reasoning_exclude": True,
        }
        self.model: ChatQwen = DefaultLangchainLLM.llm_client_factory(
            product_id="MODEL-0307",
            llm_model_name="google/gemini-2.5-flash-lite",
            user_id="92040b49",
            llm_model_config=model_config,
        )

    def test_structured_output(self):
        abstract_json_schema = {
            "title": "Document Analysis Schema",
            "description": "用于解析文档的关键信息，包括摘要、主题标签、实体、引用和术语表。",
            "type": "object",
            "properties": {
                "core_summary": {
                    "type": "string",
                    "description": "全文核心摘要，50-300字"
                },
                "theme_tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "主题标签列表"
                },
                "document_intent_type": {
                    "type": "string",
                    "description": "文档意图和类型"
                },
                "key_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "实体名称"
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "实体类型（如人名、组织、数值等）"
                            },
                            "value": {
                                "type": "string",
                                "description": "实体的具体数值或描述"
                            },
                            "context": {
                                "type": "string",
                                "description": "实体出现的上下文句子"
                            }
                        },
                        "required": ["entity_name", "entity_type", "value", "context"]
                    },
                    "description": "关键实体与数值列表"
                },
                "citation_references": {
                    "type": "object",
                    "patternProperties": {
                        "^[\\w\\d\\[\\]]+$": {
                            "type": "string",
                            "description": "引用编号到引用内容的映射"
                        }
                    },
                    "description": "引用解析字典，键为引用标记，值为引用内容"
                },
                "global_glossary": {
                    "type": "object",
                    "patternProperties": {
                        "^[\\w\\d]+$": {
                            "type": "string",
                            "description": "术语到定义的映射"
                        }
                    },
                    "description": "全局术语表，键为术语，值为定义"
                }
            },
            "required": [
                "core_summary"
            ]
        }

        system_prompt = """
你是一个专业的文档解析专家。请仔细阅读用户提供的完整文档内容，并基于全文上下文提取以下关键信息。注意：你必须通读全文才能准确理解全局主题和关联信息，避免因局部片段导致误判。

**任务要求：**
1. **全文核心摘要**：用50-300字概括文档的核心内容、主要观点和结论。
2. **主题标签**：提取3-5个代表文档主题的关键词或短语（例如："人工智能伦理"、"数据安全政策"）。
3. **文档意图/类型**：判断文档的类型和主要意图（例如：政策文件、技术报告、合同、学术论文、用户手册）。
4. **关键实体与数值**：
   - 提取文档中所有重要实体（如人名、组织、产品、地点），并标注类型和上下文中的角色（例如："王总" → "王建国，CEO"）。
   - 提取关键数值及其单位（如金额、百分比、日期），确保数值与上下文关联（例如："预算500万元"）。
   - **如果文档中不存在关键实体或数值，则返回空数组 []**
5. **引用解析**：
   - 识别所有引用标记（如[1]、[2]或"参见XX"），并提取对应的引用内容或来源（例如：将"[1]"映射到"参考文献1: XXX"）。
   - **如果文档中没有引用标记，则返回空对象 {}**
6. **全局术语表**：
   - 整理文档中出现的专业术语、缩写词及其定义（例如："RAG：检索增强生成（Retrieval-Augmented Generation）"）。
   - **如果文档中没有专业术语或缩写词，则返回空对象 {}**

**输出格式要求：**
- 必须严格使用JSON格式输出，结构如下方Schema所示。
- 所有字段均为必需，如果某个部分无内容则返回空数组或空对象。
- 实体和术语需去重，确保唯一性。

**JSON Schema定义：**
```json
{
  "type": "object",
  "properties": {
    "core_summary": {
      "type": "string",
      "description": "全文核心摘要，50-300字"
    },
    "theme_tags": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "主题标签列表"
    },
    "document_intent_type": {
      "type": "string",
      "description": "文档意图和类型"
    },
    "key_entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "entity_name": {
            "type": "string",
            "description": "实体名称"
          },
          "entity_type": {
            "type": "string",
            "description": "实体类型（如人名、组织、数值等）"
          },
          "value": {
            "type": "string",
            "description": "实体的具体数值或描述"
          },
          "context": {
            "type": "string",
            "description": "实体出现的上下文句子"
          }
        },
        "required": ["entity_name", "entity_type", "value", "context"]
      },
      "description": "关键实体与数值列表"
    },
    "citation_references": {
      "type": "object",
      "patternProperties": {
        "^[\\w\\d\\[\\]]+$": {
          "type": "string",
          "description": "引用编号到引用内容的映射"
        }
      },
      "description": "引用解析字典，键为引用标记，值为引用内容"
    },
    "global_glossary": {
      "type": "object",
      "patternProperties": {
        "^[\\w\\d]+$": {
          "type": "string",
          "description": "术语到定义的映射"
        }
      },
      "description": "全局术语表，键为术语，值为定义"
    }
  },
  "required": [
    "core_summary"
  ]
}
```

**示例输出（仅供参考）：**
```json
{
  "core_summary": "本文档详细介绍了2023年公司数据安全管理政策，强调通过加密技术和访问控制保护用户数据，并规定了违规处理流程。",
  "theme_tags": ["数据安全", "加密政策", "合规管理"],
  "document_intent_type": "政策文件",
  "key_entities": [
    {
      "entity_name": "王建国",
      "entity_type": "人名",
      "value": "CEO",
      "context": "根据王建国（CEO）的指示，所有部门需严格执行本政策。"
    },
    {
      "entity_name": "数据加密预算",
      "entity_type": "数值",
      "value": "500万元",
      "context": "2023年数据加密预算为500万元。"
    }
  ],
  "citation_references": {
    "[1]": "参考《网络安全法》第21条",
    "[2]": "参见ISO 27001标准第5.3节"
  },
  "global_glossary": {
    "RAG": "检索增强生成（Retrieval-Augmented Generation），一种结合检索和生成的AI技术",
    "GDPR": "通用数据保护条例（General Data Protection Regulation）"
  }
}
```

请根据用户给出的内容开始解析，并直接输出JSON结果。
        """

        # file_path = os.path.join(BASE_DIR, "samples", "FinDKG", "vlm", "FinDKG.md")
        file_content = ""
        with open(os.path.join(BASE_DIR, "samples", "test-html-sample.html"), 'r', encoding='utf-8') as f:
            file_content = f.read()

        file_content2 = ""
        with open(os.path.join(BASE_DIR, "samples", "LK-99-wiki-sample.html"), 'r', encoding='utf-8') as f:
            file_content2 = f.read()

        self.model.with_structured_output(abstract_json_schema,
                                          method="json_schema", )
        agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            # response_format=ProviderStrategy(schema=AbstractResponse)
        )
        # result = agent.invoke(
        #     {"messages": [
        #         {"role": "user", "content": f"{file_content}"}
        #     ]}
        # )
        #
        # print("result:", result)
        # structured_output = result.get("messages")[-1].content_blocks[-1]
        # print("structured_output:", structured_output.get("text"))
        # text_output = structured_output.get("text")
        # repaired_json = repair_json(text_output)
        # converted_json = json.loads(repaired_json)
        # print("parsed_result:", converted_json)

        results = agent.batch_as_completed(
            [
                {"messages": [
                    {"role": "user", "content": f"{file_content}"}
                ]},
                {"messages": [
                    {"role": "user", "content": f"{file_content2}"}
                ]},
            ],
            config={
                'max_concurrency': 4,  # Limit to 5 parallel calls
            }
        )
        for result in results:
            # print("result:", result)
            message = result[-1].get("messages")[-1]
            if not isinstance(message, AIMessage):
                print("[test_structured_output]-message is not AIMessage")
                continue
            json_result = repair_json(message.content, ensure_ascii=False)
            print("json_result:\n",json_result)
            print("\n\n")

    def test_structured_output1(self):
        abstract_json_schema = {
            "title": "Document Analysis Schema",
            "description": "用于解析文档的关键信息，包括摘要、主题标签、实体、引用和术语表。",
            "type": "object",
            "properties": {
                "core_summary": {
                    "type": "string",
                    "description": "全文核心摘要，50-300字"
                },
                "theme_tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "主题标签列表"
                },
                "document_intent_type": {
                    "type": "string",
                    "description": "文档意图和类型"
                },
                "key_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "实体名称"
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "实体类型（如人名、组织、数值等）"
                            },
                            "value": {
                                "type": "string",
                                "description": "实体的具体数值或描述"
                            },
                            "context": {
                                "type": "string",
                                "description": "实体出现的上下文句子"
                            }
                        },
                        "required": ["entity_name", "entity_type", "value", "context"]
                    },
                    "description": "关键实体与数值列表"
                },
                "citation_references": {
                    "type": "object",
                    "patternProperties": {
                        "^[\\w\\d\\[\\]]+$": {
                            "type": "string",
                            "description": "引用编号到引用内容的映射"
                        }
                    },
                    "description": "引用解析字典，键为引用标记，值为引用内容"
                },
                "global_glossary": {
                    "type": "object",
                    "patternProperties": {
                        "^[\\w\\d]+$": {
                            "type": "string",
                            "description": "术语到定义的映射"
                        }
                    },
                    "description": "全局术语表，键为术语，值为定义"
                }
            },
            "required": [
                "core_summary"
            ]
        }

        system_prompt = """
你是一个专业的文档解析专家。请仔细阅读用户提供的完整文档内容，并基于全文上下文提取以下关键信息。注意：你必须通读全文才能准确理解全局主题和关联信息，避免因局部片段导致误判。

**任务要求：**
1. **全文核心摘要**：用50-300字概括文档的核心内容、主要观点和结论。
2. **主题标签**：提取3-5个代表文档主题的关键词或短语（例如："人工智能伦理"、"数据安全政策"）。
3. **文档意图/类型**：判断文档的类型和主要意图（例如：政策文件、技术报告、合同、学术论文、用户手册）。
4. **关键实体与数值**：
   - 提取文档中所有重要实体（如人名、组织、产品、地点），并标注类型和上下文中的角色（例如："王总" → "王建国，CEO"）。
   - 提取关键数值及其单位（如金额、百分比、日期），确保数值与上下文关联（例如："预算500万元"）。
   - **如果文档中不存在关键实体或数值，则返回空数组 []**
5. **引用解析**：
   - 识别所有引用标记（如[1]、[2]或"参见XX"），并提取对应的引用内容或来源（例如：将"[1]"映射到"参考文献1: XXX"）。
   - **如果文档中没有引用标记，则返回空对象 {}**
6. **全局术语表**：
   - 整理文档中出现的专业术语、缩写词及其定义（例如："RAG：检索增强生成（Retrieval-Augmented Generation）"）。
   - **如果文档中没有专业术语或缩写词，则返回空对象 {}**

**输出格式要求：**
- 必须严格使用JSON格式输出，结构如下方Schema所示。
- 所有字段均为必需，如果某个部分无内容则返回空数组或空对象。
- 实体和术语需去重，确保唯一性。

**JSON Schema定义：**
```json
{
  "type": "object",
  "properties": {
    "core_summary": {
      "type": "string",
      "description": "全文核心摘要，50-300字"
    },
    "theme_tags": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "主题标签列表"
    },
    "document_intent_type": {
      "type": "string",
      "description": "文档意图和类型"
    },
    "key_entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "entity_name": {
            "type": "string",
            "description": "实体名称"
          },
          "entity_type": {
            "type": "string",
            "description": "实体类型（如人名、组织、数值等）"
          },
          "value": {
            "type": "string",
            "description": "实体的具体数值或描述"
          },
          "context": {
            "type": "string",
            "description": "实体出现的上下文句子"
          }
        },
        "required": ["entity_name", "entity_type", "value", "context"]
      },
      "description": "关键实体与数值列表"
    },
    "citation_references": {
      "type": "object",
      "patternProperties": {
        "^[\\w\\d\\[\\]]+$": {
          "type": "string",
          "description": "引用编号到引用内容的映射"
        }
      },
      "description": "引用解析字典，键为引用标记，值为引用内容"
    },
    "global_glossary": {
      "type": "object",
      "patternProperties": {
        "^[\\w\\d]+$": {
          "type": "string",
          "description": "术语到定义的映射"
        }
      },
      "description": "全局术语表，键为术语，值为定义"
    }
  },
  "required": [
    "core_summary"
  ]
}
```

**示例输出（仅供参考）：**
```json
{
  "core_summary": "本文档详细介绍了2023年公司数据安全管理政策，强调通过加密技术和访问控制保护用户数据，并规定了违规处理流程。",
  "theme_tags": ["数据安全", "加密政策", "合规管理"],
  "document_intent_type": "政策文件",
  "key_entities": [
    {
      "entity_name": "王建国",
      "entity_type": "人名",
      "value": "CEO",
      "context": "根据王建国（CEO）的指示，所有部门需严格执行本政策。"
    },
    {
      "entity_name": "数据加密预算",
      "entity_type": "数值",
      "value": "500万元",
      "context": "2023年数据加密预算为500万元。"
    }
  ],
  "citation_references": {
    "[1]": "参考《网络安全法》第21条",
    "[2]": "参见ISO 27001标准第5.3节"
  },
  "global_glossary": {
    "RAG": "检索增强生成（Retrieval-Augmented Generation），一种结合检索和生成的AI技术",
    "GDPR": "通用数据保护条例（General Data Protection Regulation）"
  }
}
```

请根据用户给出的内容开始解析，并直接输出JSON结果。
        """

        # file_path = os.path.join(BASE_DIR, "samples", "FinDKG", "vlm", "FinDKG.md")
        file_content = ""
        with open(os.path.join(BASE_DIR, "samples", "test-html-sample.html"), 'r', encoding='utf-8') as f:
            file_content = f.read()

        file_content2 = ""
        with open(os.path.join(BASE_DIR, "samples", "LK-99-wiki-sample.html"), 'r', encoding='utf-8') as f:
            file_content2 = f.read()

        self.model.with_structured_output(abstract_json_schema,
                                          method="json_schema", )
        agent = create_agent(
            model=self.model,
            # system_prompt=system_prompt,
            # response_format=ProviderStrategy(schema=AbstractResponse)
        )
        system_message = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        user_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"{file_content}",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        result = agent.invoke(
            {"messages": [
                system_message,
                user_message
            ]},
        )

        print("result:", result)
        structured_output = result.get("messages")[-1].content_blocks[-1]
        print("structured_output:", structured_output.get("text"))
        text_output = structured_output.get("text")
        repaired_json = repair_json(text_output)
        converted_json = json.loads(repaired_json)
        print("parsed_result:", converted_json)
        print("--------------------------------------------------")

        result = agent.invoke(
            {"messages": [
                system_message,
                user_message
            ]},
        )
        print("result:", result)
        structured_output = result.get("messages")[-1].content_blocks[-1]
        print("structured_output:", structured_output.get("text"))
        text_output = structured_output.get("text")
        repaired_json = repair_json(text_output)
        converted_json = json.loads(repaired_json)
        print("parsed_result:", converted_json)