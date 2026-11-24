import json
from typing import List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class BuildTaskFileInfo(BaseModel):
    """Build task file information"""
    knowledge_index_task_id: Optional[str] = Field(description="Knowledge index task ID")
    file_sync_record_id: str = Field(description="File sync record ID")
    file_name: str = Field(description="File name")
    created_at: Optional[str] = Field(description="Created at")
    knowledge_index_folder_name: Optional[str] = Field(default=None, description="Knowledge index folder name")
    file_path: Optional[str] = Field(default=None, description="File path")
    metadata: Optional[Union[str, dict]] = Field(default=None, description="Additional metadata")

    @model_validator(mode="after")
    def parse_metadata(self):
        """Parse metadata json string to dict if possible"""
        if isinstance(self.metadata, str) and self.metadata:
            try:
                self.metadata = json.loads(self.metadata)
            except json.JSONDecodeError:
                # Keep as string if parsing fails
                self.metadata = {}
        return self

class BuildTaskFilesData(BaseModel):
    """Build task files list data"""
    total: int = Field(description="Total records")
    list: List[BuildTaskFileInfo] = Field(default_factory=list, description="List of file information")
    pageNum: int = Field(description="Current page number")
    pageSize: int = Field(description="Items per page")
    size: int = Field(description="Number of records on current page")
    pages: int = Field(description="Total pages")
    prePage: int = Field(description="Previous page number")
    nextPage: int = Field(description="Next page number")
    hasPreviousPage: bool = Field(description="Has previous page")
    hasNextPage: bool = Field(description="Has next page")

    def to_dict_with_descriptions(self) -> dict:
        """Convert to dictionary with field descriptions included"""
        return {
            "data": self.model_dump(),
            "file_field_descriptions": {
                "knowledge_index_task_id": "Knowledge index task ID",
                "file_sync_record_id": "File sync record ID",
                "file_name": "File name",
                "created_at": "Created at"
            }
        }


class BuildTaskFilesResponse(BaseModel):
    """Build task files list response"""
    code: int = Field(description="Response code")
    msg: str = Field(description="Response message")
    data: Optional[BuildTaskFilesData] = Field(default=None, description="Response data")
