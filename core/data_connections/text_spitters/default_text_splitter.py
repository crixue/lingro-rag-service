from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter


class DefaultTextSplitter:

    def _init_recursive_character_text_splitter(self, **kwargs):
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=kwargs.get('encoding_name', 'cl100k_base'),
            chunk_size=kwargs.get('chunk_size', 400),
            chunk_overlap=kwargs.get('chunk_overlap', 20),
        )

    def _init_token_text_splitter(self, **kwargs):
        return TokenTextSplitter(
            **kwargs
        )

    @classmethod
    def get_text_splitter(cls, splitter_type: str = 'recursive_character', **kwargs):
        if splitter_type == 'recursive_character' or splitter_type is None:
            return cls()._init_recursive_character_text_splitter(**kwargs)
        elif splitter_type == 'token':
            return cls()._init_token_text_splitter(**kwargs)