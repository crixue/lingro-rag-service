import json
import unittest

from tools_definitions.lang import ToolSchemaDefinition


class TestToolDef(unittest.TestCase):

    def test_load_en_tool_definition(self):
        # Case 1: 加载英文工具定义并替换变量
        try:
            result1 = ToolSchemaDefinition.load_builtin_tool_schema_dict_static(
                lang="en",
                tool_name="chat_session_retriever_tool",
                variables={"uploaded_content": "web content and title is: 贷款市场报价利率（LPR）"}
            )
            print("Case 1 - EN with variables:")
            print(json.dumps(result1, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Case 1 Error: {type(e).__name__}: {e}")

    def test_load_en_tool_definition_without_variables(self):
        try:
            result3 = ToolSchemaDefinition.load_builtin_tool_schema_dict_static(
                lang="en",
                tool_name="chat_session_retriever_tool"
            )
            print("Case 3 - EN without variables:")
            print(json.dumps(result3, indent=2, ensure_ascii=False))
        except Exception as e:
            assert False, f"Case 3 Error: {type(e).__name__}: {e}"