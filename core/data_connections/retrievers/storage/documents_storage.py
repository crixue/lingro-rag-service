import json
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Sequence

from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from pydantic import BaseModel, Field, model_validator

from configs.model_config import logger
from core.common.common_util import documents_id_filling


class BaseDocumentsStorage(BaseModel):
    document_id_prefix: Optional[str] = None
    ids: Optional[List[str]] = None
    store: BaseStore = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def byte_store(self):
        return self.store

    @abstractmethod
    def save_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        pass

    @abstractmethod
    def retriever_documents(self, ids: Sequence[str]) -> List[Document]:
        pass

    @abstractmethod
    def delete_documents(self, ids: Sequence[str]) -> None:
        pass

    @abstractmethod
    def delete_storage_documents(self) -> None:
        pass


class DefaultDocumentsStorage(BaseDocumentsStorage):
    root_path: Union[str, Path] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def root_validate(cls, values):
        if values.get("root_path") is None:
            raise ValueError("Root path must be provided")
        # init store because other staff need to use it
        values["store"] = LocalFileStore(root_path=values.get("root_path"))
        return values

    def save_documents(self, documents: List[Document], require_document_id_filling=True, **kwargs) -> List[Document]:
        '''
        Default to save local file system
        '''
        if require_document_id_filling:
            documents = documents_id_filling(documents=documents, document_id_prefix=self.document_id_prefix, ids=self.ids)
        # self.store = LocalFileStore(root_path=self.root_path)
        full_docs = []
        for doc in documents:
            full_docs.append((doc.id, bytes(json.dumps(doc.model_dump(), ensure_ascii=False), "utf-8")))
        self.store.mset(full_docs)
        logger.info(f"[save_documents]-Saved the document to the rootPath: {self.root_path}")
        return documents

    def retriever_documents(self, ids: Sequence[str]) -> List[Document]:
        # self.store = LocalFileStore(root_path=self.root_path)
        results = self.store.mget(ids)
        docs = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, bytes):
                item = json.loads(item.decode("utf-8"))
                doc = Document(**item)
                docs.append(doc)

        return docs

    def delete_documents(self, ids: Sequence[str]) -> None:
        self.store = LocalFileStore(root_path=self.root_path)
        self.store.mdelete(ids)
        logger.info(f"[delete_documents]-Deleted from the rootPath: {self.root_path}. Ids:{ids} total {len(ids)} size documents have been deleted.")

    def delete_storage_documents(self) -> None:
        saved_dir = self.root_path
        if os.path.exists(saved_dir) and not os.path.isfile(saved_dir):
            shutil.rmtree(saved_dir)
            logger.info(f"[delete_storage_documents]-Dir:{saved_dir} have been deleted.")
        elif os.path.isdir(saved_dir):
            shutil.rmtree(saved_dir)
            logger.info(f"[delete_storage_documents]-Dir:{saved_dir} have been deleted.")
        else:
            logger.warning(f"[delete_storage_documents]-Dir not found: {saved_dir}")

class GlobalDocumentContextStorage(BaseDocumentsStorage):
    root_path: Union[str, Path] = Field(default=None)

    file_suffix_name: str = Field(default="_global_context")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def root_validate(cls, values):
        if values.get("root_path") is None:
            raise ValueError("Root path must be provided")
        # init store because other staff need to use it
        values["store"] = LocalFileStore(root_path=values.get("root_path"))
        return values

    def save_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        full_docs = []
        for doc in documents:
            doc_key = doc.id + self.file_suffix_name
            full_docs.append((doc_key, bytes(json.dumps(doc.model_dump(), ensure_ascii=False), "utf-8")))
            logger.info(f"[save_documents]-Saved the document to the rootPath: {self.root_path} filename:{doc_key}")
        self.store.mset(full_docs)
        return documents

    def retriever_documents(self, ids: Sequence[str]) -> List[Document]:
        modified_ids = [doc_id + self.file_suffix_name for doc_id in ids]
        results = self.store.mget(modified_ids)
        docs = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, bytes):
                item = json.loads(item.decode("utf-8"))
                doc = Document(**item)
                docs.append(doc)

        return docs

    def delete_documents(self, ids: Sequence[str]) -> None:
        modified_ids = [doc_id + self.file_suffix_name for doc_id in ids]
        self.store = LocalFileStore(root_path=self.root_path)
        self.store.mdelete(modified_ids)
        logger.info(f"[delete_documents]-Deleted from the rootPath: {self.root_path}. Ids:{ids} total {len(ids)} size documents have been deleted.")

    def delete_storage_documents(self) -> None:
        pass