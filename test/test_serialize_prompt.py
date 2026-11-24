import unittest

from core.model_io.prompts.serialization.deserialize import DefaultDeserializePrompts
from core.model_io.prompts.serialization.serialize import DefaultSerializePrompts, SerializeSystemMessageTemplatePrompts


class MyTestCase(unittest.TestCase):

    def test_serialize_default_prompt(self):
        save_dir = r"./test_data/prompt_definitions/"
        serializer = DefaultSerializePrompts(save_dir=save_dir)

        # Example for default_prompt
        default_prompt_info_dict = {
            "input_variables": ["adjective", "context"],
            "template": "Tell me a {adjective} joke about {context}."
        }
        result = serializer.serialize(prompt_type="default_prompt",
                                      prompt_info_dict=default_prompt_info_dict,
                                      file_name="default_prompt_test"
                                      )
        print(result)

    def test_serialize_chat_prompt(self):
        save_dir = r"./test_data/prompt_definitions/"
        serializer = DefaultSerializePrompts(save_dir=save_dir)

        # Example for default_prompt
        prompt_info_dict = {
            "system":{
                "template":"你是一个AI对话助手。请用中文回答用户的问题，你的回答需要基于给定的上下文和对话历史，如果你不知道答案，请不要编造答案。",
                "input_variables":[

                ]
            },
            "user":{
                "input_variables":[
                    "context",
                    "query",
                    "chat_history"
                ],
                "template":"以下是给定的上下文：\n {context}\n\n以下是对话历史：\n {chat_history}\n\n用户的问题是：{query}\n"
            }
        }
        result = serializer.serialize(prompt_type="chat_prompt",
                                      prompt_info_dict=prompt_info_dict,
                                      file_name="chat_prompt_test"
                                      )
        print(result)

    def test_serialize_few_shot_prompt(self):
        save_dir = r"./test_data/prompt_templates/default/guest0/test_default_chat_prompts"
        serializer = DefaultSerializePrompts(save_dir=save_dir)
        # Example for few_shot_prompt
        few_shot_prompt_info_dict = {
            "prefix": "你是一个AI对话助手。请用中文回答用户的问题，你的回答基于给定的信息，如果你不知道答案，请不要编造答案。",
            "input_variables": ["query", "chat_history"],
            "example_prompt_template": "问题是: {input}\n回答: {output}",
            "example_prompt_input_variables": ["input", "output"],
            "examples": [
                {"input": "台积电和三相哪一家高尖端芯片制造厂商规模最大?", "output": "台积电为最先进的处理器(cpu和gpu)提供领先的芯片制造技术。三星(Samsung)的晶圆代工部门也提供尖端芯片技术，但规模不及前者。"},
                {"input": "存储芯片供应占据亚洲市场一半及以上的芯片是哪些厂商？", "output": "三星和海力士三星和海力士加起来供应了全球一半以上的存储芯片"},
                {"input": "亚洲有哪些电子制造和装配的工厂？","output": "亚洲有鸿海（富士康母公司）、广达和纬创是全球顶尖的EMS代工厂。"}
            ],
            "suffix": "以下是对话历史：{chat_history}\n 用户问题是: {query}\nOutput:"
        }
        result = serializer.serialize(prompt_type="few_shot_prompt",
                                      prompt_info_dict=few_shot_prompt_info_dict,
                                      file_name="few_shot_prompt_test0",
                                      have_context=False
                                      )
        print(result)

    def test_serialize_few_shot_prompt_with_context(self):
        save_dir = r"./test_data/prompt_templates/default/guest0/test_default_chat_prompts"
        serializer = DefaultSerializePrompts(save_dir=save_dir)
        # Example for few_shot_prompt
        few_shot_prompt_info_dict = {
            "prefix": "你是一个AI对话助手。请用中文回答用户的问题，你的回答基于给定的标准回答、上下文信息和对话历史信息，优先使用的标准回答的信息，如果你不知道答案，请不要编造答案。以下是一些标准回答：",
            "input_variables": ["query", "chat_history", "context"],
            "example_prompt_template": "问题是: {input}\n回答: {output}",
            "example_prompt_input_variables": ["input", "output"],
            "examples": [
                {"input": "台积电和三相哪一家高尖端芯片制造厂商规模最大?", "output": "台积电为最先进的处理器(cpu和gpu)提供领先的芯片制造技术。三星(Samsung)的晶圆代工部门也提供尖端芯片技术，但规模不及前者。"},
                {"input": "存储芯片供应占据亚洲市场一半及以上的芯片是哪些厂商？", "output": "三星和海力士三星和海力士加起来供应了全球一半以上的存储芯片"},
                {"input": "亚洲有哪些电子制造和装配的工厂？","output": "亚洲有鸿海（富士康母公司）、广达和纬创是全球顶尖的EMS代工厂。"}
            ],
            "suffix": "以下是上下文信息:\n{context}\n\n以下是对话历史：\n{chat_history}\n\n 用户问题是: \n{query}\n"
        }
        result = serializer.serialize(prompt_type="few_shot_prompt",
                                      prompt_info_dict=few_shot_prompt_info_dict,
                                      file_name="few_shot_prompt_with_context_test0",
                                      have_context=True
                                      )
        print(result)

    def test_load_default_prompt(self):
        prompt_path = r"./test_data/prompt_definitions/default_prompt_test.json"
        des = DefaultDeserializePrompts()
        prompt = des.load_prompt(prompt_path)
        prompt_str = prompt.format(adjective="funny", context="monkeys")
        print(prompt_str)

    def test_load_few_shot_prompt(self):
        prompt_path = r"./test_data/prompt_definitions/few_shot_prompt_test.json"
        des = DefaultDeserializePrompts()
        prompt = des.load_prompt(prompt_path)
        prompt_str = prompt.format(adjective="funny")
        print(prompt_str)

    def test_load_few_shot_prompt(self):
        prompt_path = r"./test_data/prompt_definitions/few_shot_prompt_test.json"
        des = DefaultDeserializePrompts()
        prompt = des.load_prompt(prompt_path)
        prompt_str = prompt.format(adjective="funny")
        print(prompt_str)

    def test_serialize_SystemMessageTemplatePrompt(self):
        save_dir = r"./test_data/prompt_definitions/"
        serializer = SerializeSystemMessageTemplatePrompts(save_dir=save_dir)
        prompt_info_dict = {
            "system_prompt":
                "You are a sales assistant, and you know that the useful information: {context} \n"
                "Please answer user questions in a friendly manner as a salesperson based on the information you know. \n"
                "If you don't know, just answer: Sorry, I don't know about this question. Is there anything else I can help you with?"
        }
        result = serializer.serialize(prompt_info_dict=prompt_info_dict, file_name="message_template_prompt_test.json")
        print(result)


if __name__ == '__main__':
    unittest.main()
