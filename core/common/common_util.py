import uuid
from typing import List, Optional, Tuple, Callable

from langchain_core.documents import Document
import tiktoken


def generate_sub_document_ids(document_id_prefix: str, num_docs: int, ids: list[str] = None, ) -> Tuple[str, List[str]]:
    assert num_docs is not None, "Number of documents must be provided"
    if ids is not None:
        assert num_docs == len(ids), "Ids lengths must be the same as documents"
        if num_docs > 0:
            document_id_prefix = ids[0].split("_")[0]
        else:
            document_id_prefix = None
        return document_id_prefix, ids
    else:
        assert document_id_prefix is not None, "Document ID prefix must be provided"
        return document_id_prefix, [document_id_prefix + "_sub_doc_" + str(i) for i in range(num_docs)]


def generate_document_ids(document_id_prefix: str, num_docs: int, ids: list[str] = None, ) -> Tuple[str, List[str]]:
    assert num_docs is not None, "Number of documents must be provided"
    if ids is not None:
        assert num_docs == len(ids), "Ids lengths must be the same as documents"
        if num_docs > 0:
            document_id_prefix = ids[0].split("_")[0]
        else:
            document_id_prefix = None
        return document_id_prefix, ids
    else:
        if document_id_prefix is None:
            document_id_prefix = uuid.uuid4().hex
        return document_id_prefix, [document_id_prefix + "_" + str(i) for i in range(num_docs)]


def documents_id_filling(documents: List[Document] = None,
                         document_id_prefix: Optional[str] = None,
                         ids: Optional[List[str]] = None,
                         generate_doc_ids_func: Callable = None,
                         ) -> List[Document]:
    """
    Fill in the document IDs for the langchain documents. If the document_id_prefix is not provided, a new UUID4 will be generated.
    """
    if generate_doc_ids_func:
        document_id_prefix, ids = generate_doc_ids_func(document_id_prefix, len(documents), ids)
    else:
        document_id_prefix, ids = generate_document_ids(document_id_prefix, len(documents), ids)

    for i, doc in enumerate(documents):
        doc.id = ids[i]
        if document_id_prefix is not None:
            doc.metadata["prefix_id"] = document_id_prefix
        doc.metadata["doc_id"] = doc.id
    return documents


def token_counter(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in the given text using the specified encoding.

    Args:
        text: The text to count tokens for
        encoding_name: The name of the encoding to use (default: "cl100k_base")

    Returns:
        The number of tokens in the text
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        raise ValueError(f"Error counting tokens: {str(e)}")
