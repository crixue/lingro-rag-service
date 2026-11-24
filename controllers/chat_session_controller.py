import os
import shutil

from configs.model_config import CHAT_SESSION_TEMP_FILE_DIR_NAME, KB_ROOT_PATH
from controllers.common.common_constant import DOC_STORE_SUFFIX


class ChatSessionController:
    def delete_chat_session_indies_and_docs(self, chat_session_id: str, user_id: str):
        """
        Delete knowledge index and document store for a given chat session and user.
        Please refer to `tools_definitions/tools/retriever_tools.py`
        """
        parent_dir = os.path.join(CHAT_SESSION_TEMP_FILE_DIR_NAME, user_id, chat_session_id)
        knowledge_index_path = os.path.join(KB_ROOT_PATH, parent_dir)
        doc_store_path = os.path.join(KB_ROOT_PATH, parent_dir + DOC_STORE_SUFFIX)

        if os.path.exists(knowledge_index_path):
            shutil.rmtree(knowledge_index_path)

        if os.path.exists(doc_store_path):
            shutil.rmtree(doc_store_path)
