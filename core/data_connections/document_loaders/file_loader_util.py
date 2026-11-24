import logging
import mimetypes
import os
import re
import subprocess
from typing import List, Optional, Tuple

import docx2txt
from langchain_core.documents import Document

from configs.model_config import logger
from core.data_connections.document_loaders.file_loader_v2 import DefaultUnstructuredFileLoaderV2, \
    JSONUnstructuredFileLoaderV2
from unstructured.partition.auto import partition
from utils.common_util import count_tokens  # 新增导入


class FileTypeCovertUtil:
        @staticmethod
        def docx_to_txt(file_path: str):
            return docx2txt.process(file_path)

        @staticmethod
        def doc_to_docx(doc_file_path: str, docx_file_dir: str):
            result = subprocess.call(["soffice", "--headless", "--convert-to", "docx", doc_file_path, "--outdir", docx_file_dir])
            if result == 0:
                file_path = os.path.splitext(doc_file_path)[0] + ".docx"
                return file_path
            else:
                command = f"soffice --headless --convert-to docx {doc_file_path} --outdir {docx_file_dir}"
                logging.error(f"Convert {doc_file_path} to docx failed. Command: {command}."
                              f"Pls check your libreoffice installation and retry mannually.")
                raise Exception(f"Convert {doc_file_path} to docx failed. Command: {command}."
                                f"Pls check your libreoffice installation and retry mannually.")


class DefaultFileLoaderComponent:
    def __init__(self,
                 **kwargs):
        self.file_path = kwargs.get("file_path")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")
        self.kwargs = kwargs

    def _choice_suitable_loader_v2(self):
        # 使用 mimetypes.guess_type() 方法来猜测文件类型
        file_ext = os.path.splitext(self.file_path)[1]
        file_type, encoding = mimetypes.guess_type(self.file_path)
        # if file_ext == '.json' or file_type == "application/json":
        if file_ext == '.json' or file_ext == ".jsonl" or file_type == "application/json":
            return JSONUnstructuredFileLoaderV2(**self.kwargs)
        return DefaultUnstructuredFileLoaderV2(**self.kwargs)

    def load(self, **kwargs) -> List[Document]:
        suitable_loader = self._choice_suitable_loader_v2()
        return suitable_loader.load()

    def load_and_split(self, **kwargs) -> List[Document]:
        suitable_loader = self._choice_suitable_loader_v2()
        docs = suitable_loader.load_and_split(**kwargs)
        return docs

    def load_full_content(self, partition_kwargs: Optional[dict] = None, include_metadata: bool = True) -> Document:
        """
        使用 unstructured 解析并返回完整文本，不做任何切分。
        :param partition_kwargs: 透传给 unstructured.partition.auto.partition 的参数（如 ocr_languages 等）
        :param include_metadata: 是否返回基础元数据
        :return: 单个 Document，page_content 为整文
        """
        file_path = self.file_path

        elements = partition(filename=file_path, **(partition_kwargs or {}))
        full_text = "\n\n".join(
            s.strip() for s in (getattr(el, "text", "") for el in elements) if s and s.strip()
        )

        metadata = {
            "source": self.file_path,
            "filename": os.path.basename(self.file_path),
            "filetype": mimetypes.guess_type(self.file_path)[0],
        } if include_metadata else {}

        return Document(page_content=full_text or "", metadata=metadata)

    def load_size_of_content(self, size:int = 8000, partition_kwargs: Optional[dict] = None, include_metadata: bool = True) -> Document:
        """
        使用 unstructured 解析并返回指定大小的文本，不做任何切分。
        :param size: 返回内容的最大字符数
        :param partition_kwargs: 透传给 unstructured.partition.auto.partition 的参数（如 ocr_languages 等）
        :param include_metadata: 是否返回基础元数据
        :return: 单个 Document，page_content 为整文
        """
        file_path = self.file_path

        elements = partition(filename=file_path, **(partition_kwargs or {}))
        truncated_text = ""
        for el in elements:
            text = getattr(el, "text", "")
            if text and text.strip():
                remaining_size = size - len(truncated_text)
                if remaining_size <= 0:
                    break
                if len(text) > remaining_size:
                    truncated_text += text[:remaining_size]
                else:
                    truncated_text += text

        metadata = {
            "source": self.file_path,
            "filename": os.path.basename(self.file_path),
            "filetype": mimetypes.guess_type(self.file_path)[0],
        } if include_metadata else {}

        return Document(page_content=truncated_text or "", metadata=metadata)

    def load_tokens_of_content(
        self,
        max_tokens: int = 4000,
        page: int = 1,
        partition_kwargs: Optional[dict] = None,
        include_metadata: bool = True
    ) -> Tuple[Document, dict]:
        """
        使用 unstructured 解析并按 tokens 分页返回文本内容。

        :param max_tokens: 每页最大 token 数量
        :param page: 当前页码（从 1 开始）
        :param partition_kwargs: 透传给 unstructured.partition.auto.partition 的参数
        :param include_metadata: 是否返回基础元数据
        :return: 元组 (Document, pagination_info)，其中 pagination_info 包含分页信息
        """
        file_path = self.file_path

        # 解析文档
        elements = partition(filename=file_path, **(partition_kwargs or {}))

        # 提取所有元素的文本并计算每个元素的 tokens
        element_texts = []
        element_tokens = []
        for el in elements:
            text = getattr(el, "text", "")
            if text and text.strip():
                element_texts.append(text.strip())
                element_tokens.append(count_tokens(text.strip()))

        # 计算全文和总 tokens
        full_text = "\n\n".join(element_texts)
        total_tokens = sum(element_tokens)

        # 按元素累计方式计算分页
        page_content = ""
        current_tokens = 0
        start_element_index = 0
        end_element_index = 0

        # 计算起始元素索引（跳过前面的页）
        if page > 1:
            accumulated_tokens = 0
            for i, tokens in enumerate(element_tokens):
                accumulated_tokens += tokens
                if accumulated_tokens >= max_tokens * (page - 1):
                    start_element_index = i + 1
                    break

        # 从起始元素开始累计当前页内容
        use_char_based_fallback = False
        for i in range(start_element_index, len(element_texts)):
            element_text = element_texts[i]
            element_token_count = element_tokens[i]

            # 检查加入当前元素是否会超过限制
            if current_tokens + element_token_count <= max_tokens:
                if page_content:
                    page_content += "\n\n" + element_text
                else:
                    page_content = element_text
                current_tokens += element_token_count
                end_element_index = i
            else:
                # 当前元素会导致超限，停止累加
                break

        # 如果按元素累计导致返回空内容（单个元素超过 max_tokens），回退到字符累加方式
        if not page_content and start_element_index < len(element_texts):
            logger.warning(
                f"[load_tokens_of_content] Element-based pagination returned empty content. "
                f"Falling back to character-based pagination for page {page}."
            )
            use_char_based_fallback = True

            # 计算字符偏移量
            start_char = 0
            if page > 1:
                temp_text = ""
                for char in full_text:
                    temp_text += char
                    if count_tokens(temp_text) >= max_tokens * (page - 1):
                        start_char = len(temp_text)
                        break

            # 从 start_char 开始提取当前页内容
            remaining_text = full_text[start_char:]
            current_tokens = 0

            for char in remaining_text:
                page_content += char
                current_tokens = count_tokens(page_content)
                if current_tokens >= max_tokens:
                    break

        # 判断是否有下一页
        if use_char_based_fallback:
            # 字符累加模式：检查是否还有剩余字符
            char_offset = 0
            if page > 1:
                temp_text = ""
                for char in full_text:
                    temp_text += char
                    if count_tokens(temp_text) >= max_tokens * (page - 1):
                        char_offset = len(temp_text)
                        break
            has_next_page = char_offset + len(page_content) < len(full_text)
        else:
            # 元素累加模式：检查是否还有未处理的元素
            has_next_page = end_element_index + 1 < len(element_texts)

        # 构建元数据
        metadata = {
            "source": self.file_path,
            "filename": os.path.basename(self.file_path),
            "filetype": mimetypes.guess_type(self.file_path)[0],
            "page": page,
            "page_tokens": current_tokens,
            "total_tokens": total_tokens,
            "has_next_page": has_next_page,
            "pagination_mode": "char_based" if use_char_based_fallback else "element_based",
        } if include_metadata else {}

        # 构建分页信息
        pagination_info = {
            "current_page": page,
            "page_tokens": current_tokens,
            "total_tokens": total_tokens,
            "has_next_page": has_next_page,
            "max_tokens_per_page": max_tokens,
            "pagination_mode": "char_based" if use_char_based_fallback else "element_based",
        }

        return Document(page_content=page_content or "", metadata=metadata), pagination_info
