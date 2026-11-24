from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import Field


class RetrieverDocsToolCallBack(BaseCallbackHandler):
    additional_info: dict = Field(default={})

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if output is not None and isinstance(output, dict):
            self.additional_info = output