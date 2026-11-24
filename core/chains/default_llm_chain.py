from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseLLMOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate


class DefaultLLMChain:

    def __init__(self,
                 memory: BaseChatMemory = None,
                 prompt: BasePromptTemplate = None,
                 llm: BaseLanguageModel = None,
                 output_parser: BaseLLMOutputParser = None,
                 verbose: bool = True,
                 **kwargs
                 ) -> None:
        '''
        If serialize llm chain, the format is:
        {
          "memory": null,
          "verbose": true,
          "prompt_path": "prompt.json",
          "llm_path": "llm.json",
          "output_key": "text",
          "_type": "llm_chain"
        }
        '''
        self.llm_chain = LLMChain(memory=memory,
                                  prompt=prompt,
                                  llm=llm,
                                  output_parser=output_parser,
                                  verbose=verbose,
                                  **kwargs
                                  )