import random
import uuid
from typing import List, Optional, Callable, Iterator

import tiktoken
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from configs.model_config import CHARACTER_TEXT_SPLITTER, DEFAULT_SEPARATOR
from core.data_connections.text_spitters.default_text_splitter import DefaultTextSplitter


class BaseUnstructuredFileLoaderV2(UnstructuredLoader):
    def __init__(self, file_path: str, is_chinese_text: bool = True, max_characters: Optional[int] = None,
                 preferred_max_characters: int = None, mode: str = "elements",
                 chunking_strategy: str = "by_title", multipage_sections: bool = True,
                 combine_text_under_n_chars: Optional[int] = None, include_orig_elements: bool = False,
                 strategy: str = "auto", overlap: int = 0, overlap_all: bool = False,
                 post_processors: Optional[list[Callable[[str], str]]] = None, **kwargs):
        """
        TODO: 向用户界面提供一些参数的选择，例如：mode, chunking_strategy, strategy, overlap, overlap_all, post_processors
        :param file_path: The path to the file to be loaded.
        :param mode: "elements" or "single". You can run the loader in one of two modes: "single" and "elements".
                        If you use "single" mode, the document will be returned as a single langchain Document object.
                        If you use "elements" mode, the unstructured library will split the document into elements such as Title and NarrativeText.

        :param max_characters:  the hard maximum size for a chunk. No chunk will exceed this number of characters. A single element that by itself exceeds this size will be divided into two or more chunks using text-splitting.

        :param preferred_max_characters:  the “soft” maximum size for a chunk. A chunk that already exceeds this number of characters will not be extended, even if the next element would fit without exceeding the specified hard maximum. This can be used in conjunction with max_characters to set a “preferred” size, like “I prefer chunks of around 1000 characters, but I’d rather have a chunk of 1500 (max_characters) than resort to text-splitting”. This would be specified with (..., max_characters=1500, new_after_n_chars=1000).

        :param overlap: only when using text-splitting to break up an oversized chunk, include this number of characters from the end of the prior chunk as a prefix on the next. This can mitigate the effect of splitting the semantic unit represented by the oversized element at an arbitrary position based on text length.

        :param overlap_all:  also apply overlap between “normal” chunks, not just when text-splitting to break up an oversized element. Because normal chunks are formed from whole elements that each have a clean semantic boundary, this option may “pollute” normal chunks. You’ll need to decide based on your use-case whether this option is right for you.

        :param strategy:  The available strategies for PDFs are "auto", "hi_res", "ocr_only", and "fast".

        :param chunking_strategy: "basic" or "by_title". You can run the loader in one of two modes: "basic" and "by_title".
            # “basic” chunking strategy
            * The basic strategy combines sequential elements to maximally fill each chunk while respecting both the specified max_characters (hard-max) and new_after_n_chars (soft-max) option values.
            * A single element that by itself exceeds the hard-max is isolated (never combined with another element) and then divided into two or more chunks using text-splitting.
            * A Table element is always isolated and never combined with another element. A Table can be oversized, like any other text element, and in that case is divided into two or more TableChunk elements using text-splitting.
            * If specified, overlap is applied between chunks formed by splitting oversized elements and is also applied between other chunks when overlap_all is True.
            # “by_title” chunking strategy
            The by_title chunking strategy preserves section boundaries and optionally page boundaries as well. “Preserving” here means that a single chunk will never contain text that occurred in two different sections. When a new section starts, the existing chunk is closed and a new one started, even if the next element would fit in the prior chunk.
            In addition to the behaviors of the basic strategy above, the by_title strategy has the following behaviors:
            * Detect section headings. A Title element is considered to start a new section. When a Title element is encountered, the prior chunk is closed and a new chunk started, even if the Title element would fit in the prior chunk.
            * Respect page boundaries. Page boundaries can optionally also be respected using the multipage_sections argument. This defaults to True meaning that a page break does not start a new chunk. Setting this to False will separate elements that occur on different pages into distinct chunks.
            * Combine small sections. In certain documents, partitioning may identify a list-item or other short paragraph as a Title element even though it does not serve as a section heading. This can produce chunks substantially smaller than desired. This behavior can be mitigated using the combine_text_under_n_chars argument. This defaults to the same value as max_characters such that sequential small sections are combined to maximally fill the chunking window. Setting this to 0 will disable section combining.

        :param post_processors: A list of functions that will be applied to the text content of each chunk after it has been split. Each function should take a string as input and return a string as output. The functions will be applied in the order they are listed in the post_processors list. It can be found functions in the unstructured.cleaners.core module.
        """
        self.file_path = file_path
        assert self.file_path is not None, "file_path should not be None"
        if max_characters is None:
            self.max_characters = CHARACTER_TEXT_SPLITTER['cn_chunk_size'] if is_chinese_text \
                else CHARACTER_TEXT_SPLITTER['en_chunk_size']
        else:
            self.max_characters = max_characters
        if overlap is None:
            self.overlap = CHARACTER_TEXT_SPLITTER['cn_chunk_overlap'] if is_chinese_text \
                else CHARACTER_TEXT_SPLITTER['en_chunk_overlap']
        else:
            self.overlap = overlap
        self.is_chinese_text = is_chinese_text
        self.mode = mode
        self.chunking_strategy = chunking_strategy
        self.include_orig_elements = include_orig_elements
        self.multipage_sections = multipage_sections
        self.combine_text_under_n_chars = combine_text_under_n_chars
        if chunking_strategy == 'by_title' and self.combine_text_under_n_chars is None:
            self.combine_text_under_n_chars = self.max_characters
        self.preferred_max_characters = preferred_max_characters
        self.overlap = overlap
        self.overlap_all = overlap_all
        self.strategy = strategy
        self.post_processors = post_processors
        self.kwargs = kwargs
        self.loader = None

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load the file and yield documents.
        """
        self.loader = UnstructuredLoader(self.file_path,
                                         mode=self.mode,
                                         chunking_strategy=self.chunking_strategy,
                                         max_characters=self.max_characters,
                                         new_after_n_chars=self.preferred_max_characters,
                                         overlap=self.overlap,
                                         overlap_all=self.overlap_all,
                                         strategy=self.strategy,
                                         include_orig_elements=self.include_orig_elements,
                                         post_processors=self.post_processors,
                                         # 这两个参数用于 by_title 策略的更好效果(若底层版本支持)
                                         multipage_sections=self.multipage_sections,
                                         combine_text_under_n_chars=self.combine_text_under_n_chars,
                                         **self.kwargs)

        # 清理不需要的 metadata 字段
        unwanted_keys = [
            'emphasized_text_contents',
            'emphasized_text_tags',
            'file_directory',
            'languages',
            'element_id',
            'category'
        ]

        for doc in self.loader.lazy_load():
            for key in unwanted_keys:
                doc.metadata.pop(key, None)
            yield doc

    def preview_loaded_docs(self, preview_mode: str = "random", preview_lines: int = 9) -> list[Document]:
        """
        Preview the loaded documents.

        :param preview_mode: "random" or "all". If "random", it will randomly select a document to preview. If "all", it will preview all the loaded documents.
        :param preview_lines: The number of lines to preview for each document.
        """
        docs = self.load()
        if len(docs) <= preview_lines or preview_mode == "all":
            return docs
        # random mode select first 3, last 3 and 3 random in between
        selected_docs = [docs[0], docs[1], docs[2]] + random.sample(docs[3:-3], max(preview_lines - 6, 3)) + [docs[-3],
                                                                                                              docs[-2],
                                                                                                              docs[-1]]
        return selected_docs


class DefaultUnstructuredFileLoaderV2(BaseUnstructuredFileLoaderV2):
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)


class JSONUnstructuredFileLoaderV2(BaseUnstructuredFileLoaderV2):
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)
        self.jq_schema = kwargs.get("jq_schema")
        assert self.jq_schema is not None, "jq_schema should not be None"

        self.content_key = kwargs.get("content_key", None)
        self.metadata_func = kwargs.get("metadata_func", None)
        self.text_content = kwargs.get("text_content", True)
        self.json_lines = kwargs.get("json_lines", False)
        self.is_content_key_jq_parsable = kwargs.get("is_content_key_jq_parsable", False)
        self.chunk_size = CHARACTER_TEXT_SPLITTER['cn_chunk_size'] if self.is_chinese_text \
            else CHARACTER_TEXT_SPLITTER['en_chunk_size']
        self.chunk_overlap = CHARACTER_TEXT_SPLITTER['cn_chunk_overlap'] if self.is_chinese_text \
            else CHARACTER_TEXT_SPLITTER['en_chunk_overlap']
        self.separators = DEFAULT_SEPARATOR.get("cn") if self.is_chinese_text else DEFAULT_SEPARATOR.get("en")
        self.textsplitter = DefaultTextSplitter.get_text_splitter(
            splitter_type='recursive_character',
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator='end'
        )

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load the file and yield documents.
        """
        self.loader = JSONLoader(file_path=self.file_path, jq_schema=self.jq_schema, content_key=self.content_key,
                                 text_content=self.text_content, metadata_func=self.metadata_func,
                                 json_lines=self.json_lines)

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Filter documents based on token count
        content = ""
        metadata = None
        token_count = 0

        for index, doc in enumerate(self.loader.lazy_load()):
            if doc.page_content is None:
                continue
            token_count += len(tokenizer.encode(doc.page_content))
            if token_count < self.max_characters:
                if metadata is None:
                    metadata = doc.metadata
                content = content + doc.page_content + "\n"
            else:
                id = doc.metadata.get("id", str(uuid.uuid4()))
                metadata = doc.metadata if metadata is not None else {}
                metadata['seq_num'] = id + '-' + str(index)
                split_docs = self.textsplitter.split_documents([Document(page_content=doc.page_content, metadata=metadata)])
                for split_doc in split_docs:
                    yield split_doc
                # Reset for next batch
                content = ""
                metadata = None
                token_count = 0

        # Yield remaining content
        if content:
            refined_doc = Document(page_content=content)
            if metadata is not None:
                refined_doc.metadata = metadata
            yield refined_doc


class PDFUnstructuredFileLoaderV2(BaseUnstructuredFileLoaderV2):
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)
        # Applies the English and Swedish language pack for ocr. OCR is only applied
        # if the text is not available in the PDF.
        # from core.data_connections.document_loaders.pdf_loader import UnstructuredPaddlePDFLoader
        # self.loader = UnstructuredPaddlePDFLoader(self.file_path, **kwargs)


class ContentStringLoader(BaseUnstructuredFileLoaderV2):
    def __init__(self,
                 **kwargs):
        self.file_path = "ContentStringLoader"  # Dummy file_path
        super().__init__(file_path=self.file_path, **kwargs)
        self.content = kwargs.get("content", None)
        assert self.content is not None, "content should not be None"
        self.metadata = kwargs.get("metadata", {})
        self.separators = DEFAULT_SEPARATOR.get("cn") if self.is_chinese_text else DEFAULT_SEPARATOR.get("en")

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load the content and yield documents.
        """
        docs = [Document(page_content=self.content, metadata=self.metadata)]
        textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_characters,
            chunk_overlap=self.overlap,
            separators=self.separators,
            keep_separator='end'
        )
        split_docs = textsplitter.split_documents(docs)
        for doc in split_docs:
            yield doc
